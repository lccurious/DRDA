import json
import logging
import os
import time
from pathlib import Path

import torch
from torch.utils.data.dataloader import DataLoader


def init_model(net, restore):
    if restore is not None and os.path.exists(restore):
        net.load_state_dict(torch.load(restore))
        net.restore = True
        print("Restore model from {}".format(os.path.abspath(restore)))

    return net


def create_logger(cfg, cfg_name, phase='train'):
    """
    Create log and checkpoint output directories

    :param cfg: Config
    :param cfg_name:
    :param phase:
    :return:
    """
    root_output_dir = Path(cfg.OUTPUT_DIR)
    # set up logger
    if not root_output_dir.exists():
        print('=> creating {}'.format(root_output_dir))
        root_output_dir.mkdir(parents=True)

    dataset = cfg.DATASET.DATASET
    dataset = dataset.replace('->', '2')
    dataset = dataset.replace('/', '-')
    model = cfg.MODEL.NAME
    cfg_name = os.path.basename(cfg_name).split('.')[0]

    final_output_dir = root_output_dir / model / dataset / cfg_name

    print('=> creating {}'.format(final_output_dir))
    final_output_dir.mkdir(parents=True, exist_ok=True)

    time_str = time.strftime('%Y-%m-%d-%H-%M')
    log_file = '{}_{}_{}.log'.format(cfg_name, time_str, phase)
    final_log_file = final_output_dir / log_file
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=str(final_log_file),
                        format=head)
    logger = logging.getLogger(cfg_name)
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logger.addHandler(console)
    file_stream = logging.FileHandler(final_log_file)
    logger.addHandler(file_stream)
    dataset_root_name = cfg.DATASET.ROOT.split('/')[-1]

    tensorboard_log_dir = Path(cfg.LOG_DIR) / dataset_root_name / dataset / model / (cfg_name + '_' + phase + '_' + time_str)

    print('=> creating {}'.format(tensorboard_log_dir))
    tensorboard_log_dir.mkdir(parents=True, exist_ok=True)

    return logger, str(final_output_dir), str(tensorboard_log_dir)


def save_checkpoint(states_dict, is_best, best_metric, output_dir, num_keep=3):
    """
    .. math::
         \mathbf{w} = g \\frac{d\mathbf{v}}{\|\mathbf{v}\|}

    Example ::

        checkpoint_info_dict = {
            'latest_ckp': 'checkpoint_epoch_24.pth',
            'ckp_list': ['checkpoint_epoch_22.pth', 'checkpoint_epoch_23.pth', 'checkpoint_epoch_24.pth']
        }

    """
    filename = 'checkpoint_epoch_{}.pth'.format(states_dict['epoch'])

    checkpoint_info_path = os.path.join(output_dir, 'checkpoint.json')
    if Path(checkpoint_info_path).exists():
        with open(checkpoint_info_path, 'r') as fp:
            checkpoint_info_dict = json.load(fp=fp)
        checkpoint_info_dict['latest_ckp'] = filename
        checkpoint_info_dict['ckp_list'].append(filename)
    else:
        checkpoint_info_dict = {
            'metric': best_metric,
            'latest_ckp': filename,
            'ckp_list': [filename]
        }
    torch.save(states_dict, os.path.join(output_dir, filename))
    if is_best and best_metric > checkpoint_info_dict['metric']:
        checkpoint_info_dict['metric'] = best_metric
        torch.save(states_dict,
                   os.path.join(output_dir, 'checkpoint_best.pth'))
        print("Save best checkpoint to {}".format(os.path.join(output_dir, 'checkpoint_best.pth')))
    print("Save model to {}".format(os.path.join(output_dir, filename)))
    # delete the unused file
    while len(checkpoint_info_dict['ckp_list']) > num_keep:
        file_to_remove = os.path.join(output_dir, checkpoint_info_dict['ckp_list'][0])
        os.remove(file_to_remove)
        checkpoint_info_dict['ckp_list'] = checkpoint_info_dict['ckp_list'][1:]
    with open(checkpoint_info_path, 'w') as fp:
        json.dump(checkpoint_info_dict, fp=fp)


class ForeverDataIterator(object):
    """
    A data iterator that will never stop producing data

    """

    def __init__(self, data_loader: DataLoader):
        self.data_loader = data_loader
        self.iter = iter(self.data_loader)

    def __next__(self):
        try:
            data = next(self.iter)
        except StopIteration:
            self.iter = iter(self.data_loader)
            data = next(self.iter)
        return data

    def __len__(self):
        return len(self.data_loader)
