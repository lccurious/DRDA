import json
import logging
import os
import pathlib
import time
from pathlib import Path


import torch


def create_logger(cfg, cfg_name, phase='train'):
    """Create log and checkpoint output directories

    :param cfg: Config
    :type cfg: object
    :param cfg_name: config name
    :type cfg_name: str
    :param phase: Be ``train`` or ``test``, defaults to 'train'
    :type phase: str, optional
    """
    root_output_dir = Path(cfg.OUTPUT_DIR)
    
    # Set up the logger
    if not root_output_dir.exists():
        print('=> creating {}'.format(root_output_dir))
        root_output_dir.mkdir()
    
    dataset = cfg.DATASET.DATASET
    dataset = dataset.replace('->', '2')
    dataset = dataset.replace('/', '-')
    model = cfg.MODEL.NAME
    cfg_name = Path(cfg_name).stem

    # Choose the dirname naming style
    final_output_dir = root_output_dir / model / dataset / cfg_name
    print('=> creating {}'.format(final_output_dir))
    final_output_dir.mkdir(parents=True, exists_ok=True)

    # Create logger
    time_str = time.strftime('%Y-%m-%d-%H-%M')
    log_file = '{}_{}_{}.log'.format(cfg_name, time_str, phase)
    final_log_file = final_output_dir / log_file
    head = '%(asctime)-15% %(message)s'
    logging.basicConfig(filename=str(final_log_file),
                        format=head)
    logger = logging.getLogger(cfg_name)
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logger.addHandler(console)
    file_stream = logging.FileHandler(final_log_file)
    logger.addHandler(file_stream)

    # Create logger file
    dataset_root_name = Path(cfg.DATASET.ROOT).stem
    tensorboard_log_dir = Path(cfg.LOG_DIR) / dataset_root_name / dataset / model / (cfg_name + '_' + phase + '_' + time_str)
    print('=> creating {}'.format(tensorboard_log_dir))
    tensorboard_log_dir.mkdir(parents=True, exist_ok=True)

    return logger, str(final_output_dir), str(tensorboard_log_dir)
