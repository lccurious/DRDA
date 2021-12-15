import os
import queue
import threading

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets


def get_mnist(cfg, is_train=True, root='./data', batch_size=32, enrich=False):
    if enrich:
        pre_process = transforms.Compose([
            transforms.Grayscale(3),
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5]
            )

        ])
    else:
        pre_process = transforms.Compose([
            transforms.Resize(28),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=0.5, std=0.5
            )
        ])
    mnist_dataset = datasets.MNIST(root, is_train,
                                   transform=pre_process,
                                   download=True)
    mnist_data_loader = DataLoader(
        dataset=mnist_dataset,
        batch_size=batch_size,
        shuffle=is_train,
        pin_memory=cfg.PIN_MEMORY,
        num_workers=cfg.WORKERS
    )
    return mnist_data_loader


class DomainNetDataset(datasets.VisionDataset):
    """`DomainNet Domain Adaptation Dataset`

    """

    def __init__(self, root, image_set='train', transform=None, target_transform=None, transforms=None):
        super(DomainNetDataset, self).__init__(root, transforms, transform, target_transform)
        with open(os.path.join(root, image_set + '.txt'), 'r') as fp:
            file_names = [(os.path.join(root, x.strip().split(' ')[0]), int(x.strip().split(' ')[1]))
                          for x in fp.readlines()]
        self.samples = file_names

    @staticmethod
    def default_loader(path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def __getitem__(self, index):
        """

        :param index:
        :type index: int
        :return: (sample, target) where target is class_index of the target classs
        :rtype: tuple
        """
        path, target = self.samples[index]
        sample = self.default_loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target

    def __len__(self):
        return len(self.samples)


def get_domainnet(cfg, data_name, train=True):
    """
    Create the DomainNet dataset dataloader

    :param cfg: config instance of experiments/model/config.yml
    :type cfg: CfgNode
    :param data_name: name of sub dataset name
    :type data_name: str
    :param train: if this dataloader for training
    :type train: bool
    :return dataloader:
    :rtype dataloader: Dataloader
    """
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    image_dir = cfg.DATASET.ROOT
    if train:
        batch_size = cfg.TRAIN.BATCH_SIZE_PER_GPU
        transform = transforms.Compose([
            transforms.RandomResizedCrop(cfg.DATASET.IMAGE_SIZE, scale=(0.8, 1.25)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            normalize
        ])
        domainnet_data = DomainNetDataset(image_dir, image_set=data_name + '_train', transform=transform)
    else:
        batch_size = cfg.VAL.BATCH_SIZE_PER_GPU
        transform = transforms.Compose([
            transforms.Resize(cfg.DATASET.IMAGE_RESIZE),
            transforms.CenterCrop(cfg.DATASET.IMAGE_SIZE),
            transforms.ToTensor(),
            normalize
        ])
        domainnet_data = DomainNetDataset(image_dir, image_set=data_name + '_test', transform=transform)
    domainnet_dataloader = DataLoader(domainnet_data,
                                      batch_size=batch_size,
                                      shuffle=train,
                                      pin_memory=cfg.PIN_MEMORY,
                                      num_workers=cfg.WORKERS,
                                      drop_last=train)
    return domainnet_dataloader
