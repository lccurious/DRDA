import os
import argparse
import pprint
import shutil
import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

import __init_paths
from config.base import config, update_config
from models.base_model import BaseModel
from dataset.dataset_example import get_mnist
from utils.utils import create_logger


def train(params: dict):
    """General training scripts

    :param params: The custom training config
    :type params: dict
    """
    update_config(config, params)

    logger, final_out_dir, tb_log_dir = create_logger(
        config, params['cfg'], 'train'
    )
    logger.info(pprint.pformat(params))
    logger.info(config)

    torch.cuda.manual_seed(config.TRAIN.SEED)
    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enable = config.CUDNN.ENABLED

    if config.SAVE_CHECKPOINT:
        this_dir = os.path.dirname(__file__)
        shutil.copy(
            os.path.join(this_dir, '../lib/models/base_model.py'),
            final_out_dir
        )
        shutil.copy(
            os.path.abspath(__file__),
            final_out_dir
        )

    # Set model
    model = BaseModel(config)
    model.set_optimizer()

    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
        model.data_parallel()
    elif len(os.environ['CUDA_VISIBLE_DEVICES'].split(',')) != 1:
        model.data_parallel()
    model.cuda()

    # Set dataloader
    train_loader = get_mnist(config, batch_size=config.TRAIN.BATCH_SIZE_PER_GPU)
    
    writer_dict = {
        'writer': SummaryWriter(log_dir=tb_log_dir),
        'train_global_step': 0
    }

    for epoch in range(config.TRAIN.END_EPOCH):
        model.train(epoch, train_loader, writer_dict)
    
    writer_dict['writer'].close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Train Template machine learning project')
    parser.add_argument('--cfg', help='experiment configuration filepath',
                        default='experiments/template/template.yaml',
                        type=str)
    argparams = parser.parse_args()
    argparams = vars(argparams)
    train(argparams)
