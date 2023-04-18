import geoopt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from geomloss import SamplesLoss

from core.evaluate import accuracy
from core.function import da_coeff, AverageMeter, soft_clustering_assignment, mean_by_label, AverageCentroids
from core.loss import Entropy
from models.backbones import ResNetFcStiefelBranch


class DRDANet(object):
    def __init__(self, cfg):
        super(DRDANet, self).__init__()
        self.use_apex = cfg.USE_APEX
        self.num_classes = cfg.DATASET.NUM_CLASS
        self.num_dim = cfg.MODEL.BOTTLENECK_DIM
        self.backbone = ResNetFcStiefelBranch(cfg.MODEL.BACKBONE,
                                              cfg.MODEL.STIEFEL,
                                              use_bottleneck=True,
                                              bottleneck_dim=self.num_dim,
                                              new_cls=True,
                                              num_class=self.num_classes)
        self.use_stiefel = cfg.MODEL.STIEFEL

        self.source_gamma = cfg.MODEL.SOURCE_GAMMA
        self.criterion = nn.CrossEntropyLoss()
        self.criterion_dist = nn.MSELoss()
        self.criterion_angular = nn.CosineEmbeddingLoss()
        self.criterion_kl = nn.KLDivLoss(reduction='batchmean')
        self.criterion_sinkhorn = SamplesLoss(scaling=0.9)
        self.criterion_entropy = Entropy()
        self.moving_src = 0.0
        self.moving_tgt = 0.0
        self.moving_group_src = torch.zeros(self.num_classes, self.num_dim)
        self.moving_group_tgt = torch.zeros(self.num_classes, self.num_dim)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.max_iter = cfg.TRAIN.END_EPOCH
        self.burn_in = self.max_iter * cfg.MODEL.BURN_IN

        self.theta_move = cfg.MODEL.THETA_MOVE
        self.lambda_clf = cfg.MODEL.LAMBDA_CLF
        self.lambda_kl = cfg.MODEL.LAMBDA_KL
        self.lambda_wasserstein = cfg.MODEL.LAMBDA_WASSERSTEIN
        self.lambda_dist = cfg.MODEL.LAMBDA_DIST
        self.lambda_dist_c = cfg.MODEL.LAMBDA_DIST_C
        self.lambda_angular = cfg.MODEL.LAMBDA_ANGULAR
        self.optimizer_list = []
        self.log_dict = {}

    def cuda(self):
        self.backbone.cuda()
        self.moving_group_src = self.moving_group_src.cuda()
        self.moving_group_tgt = self.moving_group_tgt.cuda()

    def set_optimizer(self, which_opt='SGD', lr=0.001, momentum=0.9, weight_decay=0.0005):
        if which_opt == 'SGD':
            self.opt_backbone = optim.SGD(self.backbone.get_parameters(),
                                          momentum=momentum,
                                          lr=lr,
                                          weight_decay=weight_decay,
                                          nesterov=True)
            self.optimizer_list.append(self.opt_backbone)
            if self.use_stiefel:
                self.opt_stiefel = geoopt.optim.RiemannianSGD(self.backbone.get_riemann_parameters(),
                                                              momentum=momentum,
                                                              lr=lr,
                                                              weight_decay=weight_decay,
                                                              nesterov=True)
                self.optimizer_list.append(self.opt_stiefel)
        elif which_opt == 'Adam':
            self.opt_backbone = optim.Adam(self.backbone.get_parameters(),
                                           lr=lr,
                                           betas=(0.9, 0.99))
            self.optimizer_list.append(self.opt_backbone)
            if self.use_stiefel:
                self.opt_stiefel = geoopt.optim.RiemannianAdam(self.backbone.get_riemann_parameters(),
                                                               lr=lr,
                                                               betas=(0.9, 0.99))
                self.optimizer_list.append(self.opt_stiefel)

    def data_parallel(self):
        self.backbone = nn.DataParallel(self.backbone)

    def global_loss(self, gc_src, gc_tgt):
        """
        Global loss of two domains.

        Notes:
            NNI search results shows that directly minimize the distance of source and target global
            centroids to origin point would harm final performance

        :param gc_src:
        :param gc_tgt:
        :return:
        """
        loss = self.lambda_dist_c * self.criterion_dist(gc_tgt, gc_src)
        return loss

    def local_loss(self,
                   gc_src, gc_tgt,
                   anchors_src, anchors_tgt,
                   features_src=None, feature_tgt=None,
                   labels_src=None, labels_tgt=None):
        """
        local discrepancy between two domains

        :param gc_src:
        :param gc_tgt:
        :param anchors_src:
        :param anchors_tgt:
        :param features_src:
        :param feature_tgt:
        :param labels_src:
        :param labels_tgt:
        :return:
        """
        vectors_src = anchors_src - gc_src
        vectors_tgt = anchors_tgt - gc_tgt
        dist = self.criterion_dist(vectors_src, vectors_tgt)
        angular_dist = self.criterion_angular(vectors_tgt, vectors_src,
                                              target=torch.tensor(1.0, device=self.device))
        loss = self.lambda_angular * angular_dist + self.lambda_dist * dist
        return loss

    def reset_grad(self):
        for opt in self.optimizer_list:
            opt.zero_grad()

    def group_step(self, opt_group):
        for opt in opt_group:
            opt.step()
        self.reset_grad()

    def set_lr(self, iter_num, gamma, power, lr=0.01, weight_decay=0.0005):
        """
        Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs.

        :param iter_num:
        :param gamma:
        :param power:
        :param lr:
        :param weight_decay:
        :return:
        """
        lr = lr * (1 + gamma * iter_num / self.max_iter) ** (-power)
        i = 0
        for optimizer in self.optimizer_list:
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr * param_group['lr_mult']
                param_group['weight_decay'] = weight_decay * param_group['decay_mult']
                i += 1

    def compute_centroids_distance(self, source_loader, target_loader, writer_dict):
        """
        Compute the centroids in whole dataset level

        :param source_loader:
        :param target_loader:
        :param writer_dict:
        :return:
        """
        self.backbone.eval()
        centroids_src_pseudo = torch.zeros(self.num_classes, self.num_dim).cuda()
        centroids_src_true = torch.zeros(self.num_classes, self.num_dim).cuda()
        src_label_count = torch.zeros(self.num_classes).cuda()
        src_pseudo_label_count = torch.zeros(self.num_classes).cuda()
        centroids_tgt_pseudo = torch.zeros(self.num_classes, self.num_dim).cuda()
        centroids_tgt_true = torch.zeros(self.num_classes, self.num_dim).cuda()
        tgt_label_count = torch.zeros(self.num_classes).cuda()
        tgt_pseudo_label_count = torch.zeros(self.num_classes).cuda()

        centroids_src_true_m = AverageCentroids(self.num_classes, self.num_dim).cuda()
        centroids_src_pseudo_m = AverageCentroids(self.num_classes, self.num_dim).cuda()
        centroids_tgt_true_m = AverageCentroids(self.num_classes, self.num_dim).cuda()
        centroids_tgt_pseudo_m = AverageCentroids(self.num_classes, self.num_dim).cuda()

        with torch.no_grad():
            for images, labels in source_loader:
                images = images.cuda()
                features_src, logits_src = self.backbone(images)
                pseudo_src_labels = logits_src.argmax(dim=1)

                torch.zero_(src_label_count)
                torch.zero_(centroids_src_true)
                for i, label in enumerate(labels):
                    src_label_count[label] += 1
                    centroids_src_true[label] += features_src[i]
                centroids_src_true_m.update(centroids_src_true, src_label_count)

                torch.zero_(src_pseudo_label_count)
                torch.zero_(centroids_src_pseudo)
                for i, label in enumerate(pseudo_src_labels):
                    src_pseudo_label_count[label] += 1
                    centroids_src_pseudo[label] += features_src[i]
                centroids_src_pseudo_m.update(centroids_src_pseudo, src_pseudo_label_count)

            centroids_src_true = centroids_src_true_m.avg
            centroids_src_pseudo = centroids_src_pseudo_m.avg

            for images, labels in target_loader:
                images = images.cuda()
                features_tgt, logits_tgt = self.backbone(images)
                pseudo_tgt_labels = logits_tgt.argmax(dim=1)

                torch.zero_(tgt_label_count)
                torch.zero_(centroids_tgt_true)
                for i, label in enumerate(labels):
                    tgt_label_count[label] += 1
                    centroids_tgt_true[label] += features_tgt[i]
                centroids_tgt_true_m.update(centroids_tgt_true, tgt_label_count)

                torch.zero_(tgt_pseudo_label_count)
                torch.zero_(centroids_tgt_pseudo)
                for i, label in enumerate(pseudo_tgt_labels):
                    tgt_pseudo_label_count[label] += 1
                    centroids_tgt_pseudo[label] += features_tgt[i]
                centroids_tgt_pseudo_m.update(centroids_tgt_pseudo, tgt_pseudo_label_count)

            centroids_tgt_true = centroids_tgt_true_m.avg
            centroids_tgt_pseudo = centroids_tgt_pseudo_m.avg

            # compute the distances between clusters
            src_to_src_pseudo = self.local_loss(self.moving_src, self.moving_src,
                                                self.moving_group_src, centroids_src_pseudo)
            src_to_src_true = self.local_loss(self.moving_src, self.moving_src,
                                              self.moving_group_src, centroids_src_true)
            tgt_to_tgt_pseudo = self.local_loss(self.moving_tgt, self.moving_tgt,
                                                self.moving_group_tgt, centroids_tgt_pseudo)
            tgt_to_tgt_true = self.local_loss(self.moving_tgt, self.moving_tgt,
                                              self.moving_group_tgt, centroids_tgt_true)
            tgt_to_src_pseudo = self.local_loss(self.moving_tgt, self.moving_src,
                                                self.moving_group_tgt, centroids_src_pseudo)
            tgt_to_src_true = self.local_loss(self.moving_tgt, self.moving_src,
                                              self.moving_group_tgt, centroids_src_true)
            src_to_tgt_pseudo = self.local_loss(self.moving_src, self.moving_tgt,
                                                self.moving_group_src, centroids_tgt_pseudo)
            src_to_tgt_true = self.local_loss(self.moving_src, self.moving_tgt,
                                              self.moving_group_src, centroids_tgt_true)
            tgt_true_to_src_true = self.local_loss(self.moving_tgt, self.moving_src,
                                                   centroids_tgt_true, centroids_src_true)
            writer_dict['writer'].add_scalars('Loss/centroids_distance', {
                'src_to_src_pseudo': src_to_src_pseudo,
                'src_to_src_true': src_to_src_true,
                'tgt_to_tgt_pseudo': tgt_to_tgt_pseudo,
                'tgt_to_tgt_true': tgt_to_tgt_true,
                'src_to_tgt_pseudo': src_to_tgt_pseudo,
                'src_to_tgt_true': src_to_tgt_true,
                'tgt_to_src_pseudo': tgt_to_src_pseudo,
                'tgt_to_src_true': tgt_to_src_true,
                'tgt_true_to_src_true': tgt_true_to_src_true
            }, global_step=writer_dict['train_global_step'])

    def train(self, iteration, imgs_src, labels_src, imgs_tgt, writer_dict=None):
        """
        Main body for training procedure
        :param iteration:
        :param imgs_src:
        :param labels_src:
        :param imgs_tgt:
        :param labels_domain:
        :param writer_dict:
        :param vis_hist:
        :return:
        """
        self.backbone.train()
        self.reset_grad()

        # Feature extraction
        features_src, logits_src = self.backbone(imgs_src, is_tgt=False)
        features_tgt, logits_tgt = self.backbone(imgs_tgt, is_tgt=True)

        # Global feature centroid calculation
        moving_src = self.theta_move * torch.mean(features_src, dim=0) + (1 - self.theta_move) * self.moving_src
        moving_tgt = self.theta_move * torch.mean(features_tgt, dim=0) + (1 - self.theta_move) * self.moving_tgt

        prob_src = F.softmax(logits_src, dim=1)
        prob_tgt = F.softmax(logits_tgt, dim=1)
        pseudo_labels_tgt = prob_tgt.argmax(dim=1)

        # Compute new moving_group with mean
        mean_group_src = mean_by_label(features_src, labels_src, self.num_classes)
        mean_group_tgt = mean_by_label(features_tgt, pseudo_labels_tgt, self.num_classes)

        # Compute new moving_group with soft assignment
        # mean_group_src = torch.mm(F.normalize(prob_src.T, p=1, dim=1), features_src)
        # mean_group_tgt = torch.mm(F.normalize(prob_tgt.T, p=1, dim=1), features_tgt)
        moving_group_src = self.theta_move * mean_group_src + (1 - self.theta_move) * self.moving_group_src
        moving_group_tgt = self.theta_move * mean_group_tgt + (1 - self.theta_move) * self.moving_group_tgt

        # compare the moving centroids with dataset level centroids

        # Soft Assignment
        pi_src = soft_clustering_assignment(features_src, moving_group_src.detach())
        pi_tgt = soft_clustering_assignment(features_tgt, moving_group_tgt.detach())

        # Wasserstein distance from labels to anchors
        w_loss_src = self.criterion_sinkhorn(features_src, moving_group_src.detach())
        w_loss_tgt = self.criterion_sinkhorn(features_tgt, moving_group_tgt.detach())

        self.log_dict['Loss/wasserstein_source'] = w_loss_src
        self.log_dict['Loss/wasserstein_target'] = w_loss_tgt
        if self.lambda_wasserstein != 0.0 and iteration >= self.burn_in:
            w_loss = self.lambda_wasserstein * (w_loss_src + w_loss_tgt)
        else:
            w_loss = 0.0

        loss_clf = self.criterion(logits_src * self.source_gamma, labels_src)

        # Soft Assignment disagreement between classifier and clustering
        _, logits_src = self.backbone(features_src.detach() - moving_src.detach(), x_is_feature=True)
        _, logits_tgt = self.backbone(features_tgt.detach() - moving_tgt.detach(), x_is_feature=True)
        d_kl_src = self.criterion_kl(F.log_softmax(logits_src, dim=1), F.softmax(pi_src.detach(), dim=1))
        d_kl_tgt = self.criterion_kl(F.log_softmax(logits_tgt, dim=1), F.softmax(pi_tgt.detach(), dim=1))
        loss_entropy_src = self.criterion_entropy(pi_src)
        loss_entropy_tgt = self.criterion_entropy(pi_tgt)
        self.log_dict['Loss/entropy_source'] = loss_entropy_src
        self.log_dict['Loss/entropy_target'] = loss_entropy_tgt
        self.log_dict['Loss/KL_source'] = d_kl_src
        self.log_dict['Loss/KL_target'] = d_kl_tgt

        if self.lambda_kl != 0.0 and iteration >= self.burn_in:
            loss_kl = self.lambda_kl * (d_kl_src + d_kl_tgt + loss_entropy_src + loss_entropy_tgt)
        else:
            loss_kl = 0.0

        da_factor = da_coeff(iteration, max_iter=self.max_iter)

        loss_global = self.global_loss(moving_src, moving_tgt)

        # some burn in needed
        loss_local = self.local_loss(moving_src.detach(), moving_tgt.detach(),
                                     moving_group_src, moving_group_tgt,
                                     features_src, features_tgt,
                                     labels_src, pseudo_labels_tgt)
        self.log_dict['Loss/local'] = loss_local
        
        if iteration < self.burn_in:
            loss_local = 0.0

        loss = self.lambda_clf * loss_clf + da_factor * (loss_global + loss_local + w_loss + loss_kl)
        loss.backward()
        self.group_step(self.optimizer_list)

        self.log_dict['Loss/cross_entropy'] = loss_clf
        self.log_dict['Loss/global'] = loss_global

        # update the moving centroids
        self.moving_src = moving_src.data
        self.moving_tgt = moving_tgt.data
        self.moving_group_src = moving_group_src.data
        self.moving_group_tgt = moving_group_tgt.data

        for key in self.log_dict:
            writer_dict['writer'].add_scalar(key, self.log_dict[key],
                                             global_step=writer_dict['train_global_step'])
        writer_dict['train_global_step'] += 1

    def eval(self, data_loader, tgt_names, writer_dict, vis_embedding=False):
        self.backbone.eval()
        acc_dict = {}
        with torch.no_grad():
            feature_list = []
            metadata_list = []
            for dn in data_loader:
                if dn in tgt_names:
                    is_tgt = True
                else:
                    is_tgt = False
                data_iter = tqdm(data_loader[dn], desc="Val@{}".format(dn), ncols=80)
                current_avg = AverageMeter()
                for images, labels in data_iter:
                    images = images.cuda()
                    labels = labels.long().cuda()
                    if is_tgt:
                        features, logits = self.backbone(images, is_tgt=True)
                    else:
                        features, logits = self.backbone(images, is_tgt=False)
                    acc, cnt = accuracy(logits, labels)
                    current_avg.update(acc, cnt)
                    if vis_embedding:
                        feature_list.append(features)
                        metadata = [[str(v.item()) + '\t' + dn] for v in labels[:, 0]]
                        metadata_list += metadata
                acc_dict[dn] = current_avg.avg

            if vis_embedding:
                writer_dict['writer'].add_embedding(torch.cat(feature_list, dim=0),
                                                    metadata=metadata_list,
                                                    metadata_header=['Label\tDomain'],
                                                    tag='bottleneck',
                                                    global_step=writer_dict['train_global_step'])

        return acc_dict
