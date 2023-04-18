import gc
import importlib
import os
import time
from pathlib import Path

import numpy as np
from nvidia import dali
import nvidia.dali.fn as fn
import nvidia.dali.types as types
import torch
from nvidia.dali import pipeline_def, Pipeline
from nvidia.dali.plugin.pytorch import DALIClassificationIterator


def clear_memory(verbose=False):
    stt = time.time()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()  # https://forums.fast.ai/t/clearing-gpu-memory-pytorch/14637
    gc.collect()

    if verbose:
        print('Cleared memory.  Time taken was %f secs' % (time.time() - stt))


def random_transform():
    dst_cx, dst_cy = (112, 112)
    src_cx, src_cy = (112, 112)

    # This function uses homogeneous coordinates - hence, 3x3 matrix

    # translate output coordinates to center defined by (dst_cx, dst_cy)
    t1 = np.array([[1, 0, -dst_cx],
                   [0, 1, -dst_cy],
                   [0, 0, 1]])

    def u():
        return np.random.uniform(-0.2, 0.2)

    # apply a randomized affine transform - uniform scaling + some random distortion
    m = np.array([
        [1 + u(),     u(),  0],
        [    u(), 1 + u(),  0],
        [      0,       0,  1]])

    # translate input coordinates to center (src_cx, src_cy)
    t2 = np.array([[1, 0, src_cx],
                   [0, 1, src_cy],
                   [0, 0, 1]])

    # combine the transforms
    m = (np.matmul(t2, np.matmul(m, t1)))

    # remove the last row; it's not used by affine transform
    return m[0:2, 0:3].astype(np.float32)


def train_pipeline(jpegs, labels, image_resize, image_size, mean, std,
                   flip=True, random_affine=False, gaussian_sigma=0.2, rotate=0.2):
    images = fn.decoders.image(jpegs, device='mixed')
    images = fn.resize(
        images,
        size=image_resize,
        interp_type=types.INTERP_LINEAR
    )
    if flip:
        images = fn.flip(images)
    if random_affine:
        transform = fn.external_source(batch=False, source=random_transform)
        images = fn.warp_affine(images.gpu(),
                                transform.gpu(),
                                # size,        # keep the original canvas size
                                interp_type=types.INTERP_LINEAR)
    if gaussian_sigma:
        kernel_size = int(gaussian_sigma + 0.5) * 8 + 1
        images = fn.gaussian_blur(images, sigma=gaussian_sigma, window_size=kernel_size)
    if rotate:
        images = fn.rotate(images, angle=fn.random.uniform(range=(0.0, rotate)),
                           keep_size=True)
    images = fn.crop_mirror_normalize(
        images,
        dtype=types.FLOAT,
        crop=image_size,
        mean=mean,
        std=std,
        output_layout=types.NCHW
    )
    return images, labels


def val_pipeline(jpegs, labels, image_resize, image_size, mean, std):
    images = fn.decoders.image(jpegs, device='mixed')
    images = fn.resize(
        images,
        size=image_resize,
        interp_type=types.INTERP_LINEAR
    )
    images = fn.crop_mirror_normalize(
        images,
        dtype=types.FLOAT,
        crop=image_size,
        mean=mean,
        std=std
    )
    return images, labels


@pipeline_def
def file_reader_pipeline(image_dir, num_gpus, image_resize, image_size, mean, std, flip=True, rotate=0.2, train=True):
    jpegs, labels = fn.readers.file(
        file_root=image_dir,
        random_shuffle=True,
        shard_id=Pipeline.current().device_id,
        num_shards=num_gpus,
        name='Reader'
    )
    if train:
        return train_pipeline(jpegs, labels, image_resize, image_size, mean, std,
                              flip=flip, random_affine=True, gaussian_sigma=0, rotate=rotate)
    else:
        return val_pipeline(jpegs, labels, image_resize, image_size, mean, std)


class DaliIterator(object):
    def __init__(self, pipelines, size, train=True, **kwargs):
        if train:
            self._dali_iterator = DALIClassificationIterator(
                reader_name="Reader",
                pipelines=pipelines, size=size,
                last_batch_policy=0, last_batch_padded=False)
        else:
            self._dali_iterator = DALIClassificationIterator(
                reader_name="Reader",
                pipelines=pipelines, size=size,
                last_batch_policy=2, last_batch_padded=True
            )

    def __iter__(self):
        return self

    def __len__(self):
        return int(np.ceil(self._dali_iterator._size / self._dali_iterator.batch_size))


class DaliIteratorGPU(DaliIterator):
    def __next__(self):
        try:
            data = next(self._dali_iterator)
        except StopIteration:
            self._dali_iterator.reset()
            raise StopIteration

        inputs = data[0]['data']
        targets = data[0]['label'].squeeze().long()

        return inputs, targets


class DataloaderDali(object):
    def __init__(self, pipelines, id=-1):
        self.pipelines = pipelines
        self.id = id
        for pipeline in pipelines:
            pipeline.build()
        self.data_iterator = DaliIteratorGPU(pipelines, -1)

    def reset(self):
        clear_memory()
        del self.data_iterator
        clear_memory()
        importlib.reload(dali)
        from nvidia.dali.plugin.pytorch import DALIClassificationIterator
        self.data_iterator = DaliIteratorGPU(self.pipelines, self.id)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            inputs, targets = next(self.data_iterator)
        except StopIteration:
            self.reset()
            raise StopIteration

        return inputs, targets


class Dataset(object):
    def __init__(self, cfg, train=True):
        self.is_train = train
        self.dataset_name = Path(cfg.DATASET.ROOT).stem
        self.dataset_root = cfg.DATASET.ROOT
        self.num_workers = cfg.WORKERS
        self.image_resize = cfg.DATASET.IMAGE_RESIZE
        self.image_size = cfg.DATASET.IMAGE_SIZE
        self.mean = cfg.DATASET.MEAN
        self.std = cfg.DATASET.STD
        self.batch_size = cfg.TRAIN.BATCH_SIZE_PER_GPU
        self.num_gpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
        self.dataset_list_src = cfg.DATASET.DATASET.split('->')[0].split('/')
        self.dataset_list_tgt = cfg.DATASET.DATASET.split('->')[1].split('/')
        self.src_dataloader = {}
        self.tgt_dataloader = {}

    def _build_pipeline(self, val_on_cpu=True):
        src_batch_size_mode = self.batch_size % len(self.dataset_list_src)
        add_mode_src = True
        for dn in self.dataset_list_src:
            if self.dataset_name == 'Office31':
                image_dir = str(Path(self.dataset_root) / dn / 'images')
            else:
                image_dir = str(Path(self.dataset_root) / dn)

            if add_mode_src:
                this_batch_size = self.batch_size // len(self.dataset_list_src) + src_batch_size_mode
                add_mode_src = False
            else:
                this_batch_size = self.batch_size // len(self.dataset_list_src)

            pipelines = [
                file_reader_pipeline(
                    batch_size=this_batch_size,
                    num_threads=self.num_workers,
                    device_id=i,
                    num_gpus=self.num_gpus,
                    image_dir=image_dir,
                    image_resize=self.image_resize,
                    image_size=self.image_size,
                    mean=self.mean,
                    std=self.std,
                    train=self.is_train
                ) for i in range(self.num_gpus)
            ]
            for pipeline in pipelines:
                pipeline.build()
            self.src_dataloader[dn] = DataloaderDali(pipelines, -1)

        tgt_batch_size_mode = self.batch_size % len(self.dataset_list_tgt)
        add_mode_tgt = True
        for dn in self.dataset_list_tgt:
            if self.dataset_name == 'Office31':
                image_dir = str(Path(self.dataset_root) / dn / 'images')
            else:
                image_dir = str(Path(self.dataset_root) / dn)

            if add_mode_tgt:
                this_batch_size = self.batch_size // len(self.dataset_list_tgt) + tgt_batch_size_mode
                add_mode_tgt = False
            else:
                this_batch_size = self.batch_size // len(self.dataset_list_tgt)

            pipelines = [
                file_reader_pipeline(
                    batch_size=this_batch_size,
                    num_threads=self.num_workers,
                    device_id=i,
                    num_gpus=self.num_gpus,
                    image_dir=image_dir,
                    image_resize=self.image_resize,
                    image_size=self.image_size,
                    mean=self.mean,
                    std=self.std,
                    train=self.is_train
                ) for i in range(self.num_gpus)
            ]
            for pipeline in pipelines:
                pipeline.build()
            self.tgt_dataloader[dn] = DataloaderDali(pipelines, -1)

    def reset(self):
        # This is needed only for DALI
        if self.use_dali:
            # Currently we need to delete & rebuild the dali pipeline every epoch,
            # due to a memory leak somewhere in DALI
            # print('Recreating DALI dataloaders to reduce memory usage')
            del self.train_loader, self.val_loader, self.train_pipe, self.val_pipe
            clear_memory()

            # taken from: https://stackoverflow.com/questions/1254370/reimport-a-module-in-python-while-interactive
            importlib.reload(dali)
            from nvidia.dali.plugin.pytorch import DALIClassificationIterator
            # from nvidia.dali import HybridTrainPipe, HybridValPipe, DaliIteratorCPU, DaliIteratorGPU

            self._build_pipeline(val_on_cpu=False)


def get_dali_dataloader(cfg, extra_data_name=None, train=True):
    src_dataset_list = cfg.DATASET.DATASET.split('->')[0].split('/')
    tgt_dataset_list = cfg.DATASET.DATASET.split('->')[1].split('/')

    src_dataloader = {}
    tgt_dataloader = {}

    support_datasets = {
        'Office31': ['amazon', 'dslr', 'webcam'],
        'OfficeHome': ['Art', 'Clipart', 'Product', 'Real_World'],
        'visda': ['visda17train', 'visda17validation'],
        'Domainnet': ['quickdraw', 'clipart', 'sketch', 'painting', 'real', 'infograph']
    }

    dataset_name = Path(cfg.DATASET.ROOT).stem
    image_resize = cfg.DATASET.IMAGE_RESIZE
    image_size = cfg.DATASET.IMAGE_SIZE
    mean = cfg.DATASET.MEAN
    std = cfg.DATASET.STD
    batch_size = cfg.TRAIN.BATCH_SIZE_PER_GPU
    num_gpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    # load source dataset
    src_batch_size_mode = batch_size % len(src_dataset_list)
    add_mode_src = True

    if extra_data_name is not None:
        for dn in extra_data_name.split('/'):
            if dataset_name in ['Office31', 'office_caltech_10']:
                image_dir = str(Path(cfg.DATASET.ROOT) / dn / 'images')
            else:
                image_dir = str(Path(cfg.DATASET.ROOT) / dn)

            pipelines = [
                file_reader_pipeline(
                    batch_size=batch_size,
                    num_threads=cfg.WORKERS,
                    device_id=i,
                    num_gpus=num_gpus,
                    image_dir=image_dir,
                    image_resize=image_resize,
                    image_size=image_size,
                    mean=mean,
                    std=std,
                    train=train
                ) for i in range(num_gpus)
            ]
            for pipeline in pipelines:
                pipeline.build()
            tgt_dataloader[dn] = DataloaderDali(pipelines, -1)
        return tgt_dataloader

    for dn in src_dataset_list:
        if dataset_name in ['Office31', 'office_caltech_10']:
            image_dir = str(Path(cfg.DATASET.ROOT) / dn / 'images')
        else:
            image_dir = str(Path(cfg.DATASET.ROOT) / dn)

        if add_mode_src:
            this_batch_size = batch_size // len(src_dataset_list) + src_batch_size_mode
            add_mode_src = False
        else:
            this_batch_size = batch_size // len(src_dataset_list)

        pipelines = [
            file_reader_pipeline(
                batch_size=this_batch_size,
                num_threads=cfg.WORKERS,
                device_id=i,
                num_gpus=num_gpus,
                image_dir=image_dir,
                image_resize=image_resize,
                image_size=image_size,
                mean=mean,
                std=std,
                train=train
            ) for i in range(num_gpus)
        ]
        for pipeline in pipelines:
            pipeline.build()
        src_dataloader[dn] = DataloaderDali(pipelines, -1)

    tgt_batch_size_mode = batch_size % len(tgt_dataset_list)
    add_mode_tgt = True
    for dn in tgt_dataset_list:
        if dataset_name in ['Office31', 'office_caltech_10']:
            image_dir = str(Path(cfg.DATASET.ROOT) / dn / 'images')
        else:
            image_dir = str(Path(cfg.DATASET.ROOT) / dn)

        if add_mode_tgt:
            this_batch_size = batch_size // len(tgt_dataset_list) + tgt_batch_size_mode
            add_mode_tgt = False
        else:
            this_batch_size = batch_size // len(tgt_dataset_list)

        pipelines = [
            file_reader_pipeline(
                batch_size=this_batch_size,
                num_threads=cfg.WORKERS,
                device_id=i,
                num_gpus=num_gpus,
                image_dir=image_dir,
                image_resize=image_resize,
                image_size=image_size,
                mean=mean,
                std=std,
                train=train
            ) for i in range(num_gpus)
        ]
        for pipeline in pipelines:
            pipeline.build()
        tgt_dataloader[dn] = DataloaderDali(pipelines, -1)

    return src_dataloader, tgt_dataloader


if __name__ == '__main__':
    from tqdm import tqdm
    image_dir = '/path/to/dataset'
    IMAGE_RESIZE = (256, 256)
    IMAGE_SIZE = (224, 224)
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    num_gpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))

    train_pipes = [
        file_reader_pipeline(
            batch_size=64, num_threads=2, device_id=i, num_gpus=num_gpus,
            image_dir=image_dir,
            image_resize=IMAGE_RESIZE,
            image_size=IMAGE_SIZE,
            mean=MEAN,
            std=STD) for i in range(num_gpus)]
    dali_iter = DaliIteratorGPU(train_pipes, -1)

    for images, labels in tqdm(dali_iter, ncols=80):
        pass
