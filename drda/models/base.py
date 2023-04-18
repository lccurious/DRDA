import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from core.evaluate import accuracy
from core.function import AverageMeter


class BaseModel(object):
    """
    Main class for managing the model Initialization, Training Strategy and Evaluation
    """

    def __init__(self, config):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.config = config
        self.num_classes = config.DATASET.NUM_CLASS
        self.max_iter = config.TRAIN.END_EPOCH
        self.backbone = None

        self.source_gamma = config.MODEL.SOURCE_GAMMA
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer_list = []

    def cuda(self):
        self.backbone.cuda()

    def data_parallel(self):
        self.backbone = nn.DataParallel(self.backbone)

    def set_optimizer(self, which_opt='SGD', lr=0.001, momentum=0.9, weight_decay=0.0005):
        if which_opt == 'SGD':
            self.opt_backbone = optim.SGD(self.backbone.get_parameters(), momentum=momentum,
                                          lr=lr, weight_decay=weight_decay, nesterov=True)
            self.optimizer_list.append(self.opt_backbone)

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

    def train(self, iteration, imgs_src, labels_src, imgs_tgt, labels_domain=None, writer_dict=None, vis_hist=False):
        raise NotImplementedError

    def eval(self, data_loader, writer_dict, vis_embedding=False):
        self.backbone.eval()
        acc_dict = {}
        with torch.no_grad():
            feature_list = []
            metadata_list = []
            for dn in data_loader:
                data_iter = tqdm(data_loader[dn], desc="Val@{}".format(dn), ncols=80)
                current_avg = AverageMeter()
                for item in data_iter:
                    images = item[0].cuda()
                    labels = item[1].long().cuda()
                    feature, logits = self.backbone(images)
                    acc, cnt = accuracy(logits, labels)
                    current_avg.update(acc, cnt)
                    if vis_embedding:
                        feature_list.append(feature)
                        metadata = [[str(v.item()) + '\t' + dn] for v in item[0]['label'][:, 0]]
                        metadata_list += metadata
                acc_dict[dn] = current_avg.avg
            if vis_embedding:
                writer_dict['writer'].add_embedding(torch.cat(feature_list, dim=0),
                                                    metadata=metadata_list,
                                                    metadata_header=['Label\tDomain'],
                                                    tag='bottleneck',
                                                    global_step=writer_dict['train_global_step'])

        return acc_dict
