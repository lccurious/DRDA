import os
import pprint
import shutil
import argparse

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import __init_path

from config.base import config, update_config
from models.drda import DRDANet
from dataset.dataset import get_dataloader
from utils.utils import create_logger, save_checkpoint, ForeverDataIterator


def train(params: dict, save_state=False):
    """
    General training script for project

    :param params: the training config
    :param save_state: if save the checkpoint and model file into file
    """
    update_config(config, params)

    logger, final_out_dir, tb_log_dir = create_logger(
        config, params['cfg'], 'train'
    )
    writer_dict = {
        'writer': SummaryWriter(log_dir=tb_log_dir),
        'train_global_step': 0
    }
    logger.info(pprint.pformat(params))
    logger.info(config)
    val_freq = config.VAL.FRQ

    torch.cuda.manual_seed(config.TRAIN.SEED)
    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enable = config.CUDNN.ENABLED

    if save_state:
        this_dir = os.path.dirname(__file__)
        shutil.copy(
            os.path.join(this_dir, '../drda/models/drda.py'),
            final_out_dir
        )
        shutil.copy(
            os.path.abspath(__file__),
            final_out_dir
        )

    # training set dataloader
    src_train_loader, tgt_train_loader = get_dataloader(cfg=config, train=True)
    src_val_loader, tgt_val_loader = get_dataloader(cfg=config, train=False)

    model = DRDANet(config)
    model.set_optimizer(which_opt=config.TRAIN.OPTIM, lr=config.TRAIN.LR)

    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
        model.data_parallel()
    elif len(os.environ['CUDA_VISIBLE_DEVICES'].split(',')) != 1:
        model.data_parallel()
    model.cuda()

    val_dataloader = {**src_val_loader, **tgt_val_loader}

    best_metric = 0.0
    best_acc = {}
    for dn in val_dataloader:
        best_acc[dn] = 0.0
    task_info = {
        "task": config.DATASET.DATASET,
    }

    # prepare training dataloader
    src_train_data_iter, tgt_train_data_iter = {}, {}
    for dn in src_train_loader:
        src_train_data_iter[dn] = ForeverDataIterator(src_train_loader[dn])
    for dn in tgt_train_loader:
        tgt_train_data_iter[dn] = ForeverDataIterator(tgt_train_loader[dn])

    for epoch in tqdm(range(1, config.TRAIN.END_EPOCH + 1), desc='Training', ncols=80):
        model.set_lr(epoch, config.TRAIN.LR_GAMMA, 0.75, config.TRAIN.LR, config.TRAIN.WEIGHT_DECAY)
        if epoch % val_freq == 0 or epoch == config.TRAIN.END_EPOCH:
            val_acc = model.eval(val_dataloader, tgt_train_data_iter.keys(), writer_dict, vis_embedding=False)
            task_info['final'] = val_acc
            task_info['iteration'] = epoch
            model.compute_centroids_distance(list(src_train_loader.values())[0],
                                             list(tgt_train_loader.values())[0], writer_dict)
            is_best = False
            for dn in val_acc:
                writer_dict['writer'].add_scalar('val/{}'.format(dn),
                                                 val_acc[dn],
                                                 writer_dict['train_global_step'])

                if best_acc[dn] < val_acc[dn]:
                    best_acc[dn] = val_acc[dn]
                    if dn in tgt_val_loader:
                        is_best = True
                        best_metric = float(val_acc[dn])

            state_dict = {
                'epoch': epoch,
                'state_dict': model.backbone.state_dict(),
                'accuracy': val_acc
            }
            if save_state:
                save_checkpoint(state_dict, is_best, best_metric, final_out_dir)

            acc_report = ["Accuracy@{}:{:.5f}".format(
                dn, val_acc[dn]) for dn in val_acc]
            tqdm.write("Latest Val:" + "\t".join(acc_report))

        src_images, src_labels = [], []
        tgt_images, tgt_labels = [], []
        for dn in src_train_data_iter:
            images, labels = next(src_train_data_iter[dn])
            src_images.append(images)
            src_labels.append(labels)

        for dn in tgt_train_data_iter:
            images, labels = next(tgt_train_data_iter[dn])
            tgt_images.append(images)
            tgt_labels.append(labels)

        images_src = torch.cat(src_images, dim=0).cuda()
        images_tgt = torch.cat(tgt_images, dim=0).cuda()
        labels_src = torch.cat(src_labels, dim=0).cuda()

        model.train(epoch, images_src, labels_src, images_tgt, writer_dict=writer_dict)

    writer_dict['writer'].close()
    task_info['best'] = best_acc
    return task_info


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train Reason Domain Separation')
    parser.add_argument('--cfg', help="experiment configuration filename",
                        default="experiments/drda/officehome/base.yaml",
                        type=str)
    parser.add_argument('--dataset', help="dataset for training",
                        type=str, required=False)
    parser.add_argument('--logdir', help="tensorboard log dir",
                        type=str, required=False)
    parser.add_argument('--num_epoch', help="number of total iteration",
                        type=int, required=False)
    argparams = parser.parse_args()
    argparams = vars(argparams)
    shortcut_dict = {
        "cfg": "cfg",
        "logdir": "LOG_DIR",
        "num_epoch": "TRAIN.END_EPOCH",
        "dataset": "DATASET.DATASET",
        "lr": "TRAIN.LR",
        "burn_in": "MODEL.BURN_IN",
        "theta": "MODEL.THETA_MOVE",
        "lambda_clf": "MODEL.LAMBDA_CLF",
        "lambda_dist_c": "MODEL.LAMBDA_DIST_C",
        "lambda_kl": "MODEL.LAMBDA_KL",
        "lambda_dist": "MODEL.LAMBDA_DIST",
        "lambda_adv": "MODEL.LAMBDA_ADV",
        "lambda_angular": "MODEL.LAMBDA_ANGULAR",
        "lambda_wasserstein": "MODEL.LAMBDA_WASSERSTEIN",
        "gst_join_epoch": "MODEL.GST_JOIN_EPOCH",
        "temperature": "MODEL.SOURCE_GAMMA",
        "cluster_inc_freq": "TRAIN.CLUSTER_INC_FRQ"
    }
    params = {}
    for key in argparams:
        if argparams[key] is not None:
            params[shortcut_dict[key]] = argparams[key]
    train(params)
