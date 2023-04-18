import os
import queue
import threading

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
from dataset.dali_reader import get_dali_dataloader

__all__ = [
    'get_digits5',
    'get_office',
    'get_visda',
    'get_domainnet',
    'get_dataloader',
]


class BackgroundGenerator(threading.Thread):
    def __init__(self, generator, max_prefetch=1):
        """
        This function transforms generator into a background-thead generator.

        :param generator:
        :param max_prefetch:
        """
        threading.Thread.__init__(self)
        self.queue = queue.Queue(max_prefetch)
        self.generator = generator
        self.daemon = True
        self.start()

    def run(self):
        for item in self.generator:
            self.queue.put(item)
        self.queue.put(None)

    def next(self):
        next_item = self.queue.get()
        if next_item is None:
            raise StopIteration
        return next_item

    def __next__(self):
        return self.next()

    def __iter__(self):
        return self


class background:
    """
    A decorator

    @background()
    def iterate_minibatches(some_param):
        while True:
            X = read_heavy_file()
            X = do_helluva_math(X)
            y = wget_from_pornhub()
            do_pretty_much_anything()
            yield X_batch, y_batch
    """

    def __init__(self, max_prefetch=1):
        """

        :param max_prefetch:
        """
        self.max_prefetch = max_prefetch

    def __call__(self, gen):
        def bg_generator(*args, **kwargs):
            return BackgroundGenerator(gen(*args, **kwargs), max_prefetch=self.max_prefetch)

        return bg_generator


def default_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


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


def get_fashion_mnist(cfg, is_train=True, root='./data', batch_size=32):
    pre_process = transforms.Compose([
        transforms.Grayscale(3),
        transforms.Pad(2),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )
    ])
    fashion_mnist = datasets.FashionMNIST(root, is_train,
                                          transform=pre_process,
                                          download=True)
    fashion_mnist_data_loader = DataLoader(
        dataset=fashion_mnist,
        batch_size=batch_size,
        shuffle=is_train,
        pin_memory=cfg.PIN_MEMORY,
        num_workers=cfg.WORKERS
    )
    return fashion_mnist_data_loader


def get_svhn(cfg, is_train=True, root='./data/svhn', batch_size=32, enrich=True):
    if enrich:
        pre_process = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5]
            )
        ])
    else:
        pre_process = transforms.Compose([
            transforms.Grayscale(3),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
            )
        ])
    if is_train:
        train_str = 'train'
    else:
        train_str = 'test'
    svhn = datasets.SVHN(root, train_str,
                         transform=pre_process,
                         download=True)
    svhn_data_loader = DataLoader(
        dataset=svhn,
        batch_size=batch_size,
        shuffle=is_train,
        pin_memory=cfg.PIN_MEMORY,
        num_workers=cfg.WORKERS
    )
    return svhn_data_loader


def get_usps(cfg, is_train=True, root='./data', batch_size=32, enrich=False):
    if enrich:
        pre_process = transforms.Compose([
            transforms.Grayscale(3),
            transforms.Pad(2),
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
    usps = datasets.USPS(root, is_train,
                         transform=pre_process,
                         download=True)
    usps_data_loader = DataLoader(
        dataset=usps,
        batch_size=batch_size,
        shuffle=is_train,
        pin_memory=cfg.PIN_MEMORY,
        num_workers=cfg.WORKERS
    )
    return usps_data_loader


def get_digits5(cfg, data_name, train=True):
    datasets_names = cfg.DATASET.DATASET
    src_datasets = datasets_names.split('->')[0].split('/')
    tgt_datasets = datasets_names.split('->')[1].split('/')
    dataset_opts = {
        'mnist': get_mnist,
        'svhn': get_svhn,
        'usps': get_usps
    }
    if 'svhn' in datasets_names:
        return dataset_opts[data_name](cfg, train, enrich=True)
    else:
        return dataset_opts[data_name](cfg, train, enrich=False)


def get_office(cfg, data_name, train=True):
    """
    Create pytorch dataloader for Office31 Dataset and OfficeHome Dataset

    :param cfg: config from CfgNode style
    :type cfg: object
    :param data_name: name of target data sequence folder
    :type data_name: str
    :param train: is this dataloader for training
    :type train: bool
    :return dataloader: pytorch dataloader
    :rtype dataloader: Dataloader
    """
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    if data_name in ['amazon', 'dslr', 'webcam', 'caltech']:
        image_dir = cfg.DATASET.ROOT + '/' + data_name + '/images'
    else:
        image_dir = cfg.DATASET.ROOT + '/' + data_name
    if train:
        batch_size = cfg.TRAIN.BATCH_SIZE_PER_GPU
        transform = transforms.Compose([
            transforms.RandomResizedCrop(cfg.DATASET.IMAGE_SIZE, scale=(0.8, 1.25)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            normalize
        ])
        office_data = datasets.ImageFolder(root=image_dir,
                                           transform=transform)
    else:
        batch_size = cfg.VAL.BATCH_SIZE_PER_GPU
        transform = transforms.Compose([
            transforms.Resize(cfg.DATASET.IMAGE_RESIZE),
            transforms.CenterCrop(cfg.DATASET.IMAGE_SIZE),
            transforms.ToTensor(),
            normalize
        ])
        office_data = datasets.ImageFolder(root=image_dir,
                                           transform=transform)
    office_dataloader = DataLoader(office_data,
                                   batch_size=batch_size,
                                   shuffle=train,
                                   pin_memory=cfg.PIN_MEMORY,
                                   num_workers=cfg.WORKERS,
                                   drop_last=train)
    return office_dataloader


def get_visda(cfg, data_name, train=True):
    """
    Create the VisDA dataloader

    :param cfg: config instance of experiments/model/config.yml
    :type cfg: CfgNode
    :param data_name: name of dataset
    :type data_name: str
    :param train: if is for training
    :type train: bool
    :return:
    """
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    image_dir = cfg.DATASET.ROOT + '/' + data_name
    if train:
        batch_size = cfg.TRAIN.BATCH_SIZE_PER_GPU
        transform = transforms.Compose([
            transforms.RandomResizedCrop(cfg.DATASET.IMAGE_SIZE, scale=(0.8, 1.25)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            normalize
        ])
        visda_data = datasets.ImageFolder(root=image_dir,
                                          transform=transform)
    else:
        batch_size = cfg.VAL.BATCH_SIZE_PER_GPU
        transform = transforms.Compose([
            transforms.Resize(cfg.DATASET.IMAGE_RESIZE),
            transforms.CenterCrop(cfg.DATASET.IMAGE_SIZE),
            transforms.ToTensor(),
            normalize
        ])
        visda_data = datasets.ImageFolder(root=image_dir,
                                          transform=transform)
    visda_dataloader = DataLoader(visda_data,
                                  batch_size=batch_size,
                                  shuffle=train,
                                  pin_memory=cfg.PIN_MEMORY,
                                  num_workers=cfg.WORKERS,
                                  drop_last=train)
    return visda_dataloader


class DomainNetDataset(datasets.VisionDataset):
    """`DomainNet Domain Adaptation Dataset`

    """

    def __init__(self, root, image_set='train', transform=None, target_transform=None, transforms=None):
        super(DomainNetDataset, self).__init__(root, transforms, transform, target_transform)
        with open(os.path.join(root, image_set + '.txt'), 'r') as fp:
            file_names = [(os.path.join(root, x.strip().split(' ')[0]), int(x.strip().split(' ')[1]))
                          for x in fp.readlines()]
        self.samples = file_names

    def __getitem__(self, index):
        """

        :param index:
        :type index: int
        :return: (sample, target) where target is class_index of the target classs
        :rtype: tuple
        """
        path, target = self.samples[index]
        sample = default_loader(path)
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


def get_dataloader(cfg, extra_data_name=None, train=True):

    if cfg.DATASET.USE_DALI:
        return get_dali_dataloader(cfg, extra_data_name=extra_data_name, train=train)

    src_dataset_list = cfg.DATASET.DATASET.split('->')[0].split('/')
    tgt_dataset_list = cfg.DATASET.DATASET.split('->')[1].split('/')

    src_dataloader = {}
    tgt_dataloader = {}

    dn_dict = {
        'amazon': get_office,
        'caltech': get_office,
        'dslr': get_office,
        'webcam': get_office,
        'Art': get_office,
        'Clipart': get_office,
        'Product': get_office,
        'Real_World': get_office,
        'mnist': get_digits5,
        'svhn': get_digits5,
        'usps': get_digits5,
        'visda17train': get_visda,
        'visda17validation': get_visda,
        'visda17test': get_visda,
        'clipart': get_domainnet,
        'infograph': get_domainnet,
        'painting': get_domainnet,
        'quickdraw': get_domainnet,
        'real': get_domainnet,
        'sketch': get_domainnet
    }

    if extra_data_name is None:
        # load source dataset
        for dn in src_dataset_list:
            src_dataloader[dn] = dn_dict[dn](cfg, data_name=dn, train=train)

        for dn in tgt_dataset_list:
            tgt_dataloader[dn] = dn_dict[dn](cfg, data_name=dn, train=train)

        return src_dataloader, tgt_dataloader
    else:
        extra_dataloader = {}
        for dn in extra_data_name.split('/'):
            extra_dataloader[dn] = dn_dict[dn](cfg, data_name=dn, train=train)
        return extra_dataloader


class IndicateData(object):
    """
    retain a specific class of data
    """

    def __init__(self, cfg, train=False):
        self.src_loader, self.tgt_loader = get_dataloader(cfg, train)
        self.tgt_dataset_status = {}
        self.src_dataset_status = {}

    def preprocess(self):
        # TODO: statistic the dataset status {class: number}
        pass

    def pure_data(self, class_name):
        """
        return the specific data

        :param class_name:
        :return:
        """
        src_images, src_labels = [], []
        src_cnt = 0
        for dn in self.src_loader:
            for images_s, labels_s in self.src_loader[dn]:
                b = labels_s == class_name
                if b.long().sum() > 0:
                    src_images.append(images_s[b])
                    src_labels.append(labels_s[b])
                    src_cnt += torch.sum(b.long())
                if src_cnt > 32:
                    break
        images_src = torch.cat(src_images)
        labels_src = torch.cat(src_labels)

        tgt_images, tgt_labels = [], []
        tgt_cnt = 0
        for dn in self.tgt_loader:
            for images_t, labels_t in self.tgt_loader[dn]:
                b = labels_t == class_name
                if b.long().sum() > 0:
                    tgt_images.append(images_t[b])
                    tgt_labels.append(labels_t[b])
                    tgt_cnt += torch.sum(b.long())
                if tgt_cnt > 32:
                    break
        images_tgt = torch.cat(tgt_images)
        labels_tgt = torch.cat(tgt_labels)

        align_num = min(min(images_src.size(0), images_tgt.size(0)), 32)
        return images_src[:align_num], labels_src[:align_num], images_tgt[:align_num], labels_tgt[:align_num]


class ConcatDataset(Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        return tuple(d[i % len(d)] for d in self.datasets)

    def __len__(self):
        return max(len(d) for d in self.datasets)
