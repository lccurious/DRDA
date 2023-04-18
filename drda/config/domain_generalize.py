from yacs.config import CfgNode

_C = CfgNode()

# General config
_C.OUTPUT = './results'
_C.OUTPUT_DIR = './output'
_C.LOG_DIR = './log'
_C.WORKERS = 4
_C.AUTO_RESUME = False
_C.PIN_MEMORY = True
_C.RANK = 0
_C.USE_APEX = False

# CUDNN related params
_C.CUDNN = CfgNode()
_C.CUDNN.BENCHMARK = True
_C.CUDNN.DETERMINISTIC = False
_C.CUDNN.ENABLED = True

# Dataset
_C.DATASET = CfgNode()
_C.DATASET.USE_DALI = False
_C.DATASET.DATASET = ''  # will be update later
_C.DATASET.EXTERNAL_DATASET = None  # will be update later
_C.DATASET.MEAN = [0.485, 0.456, 0.406]
_C.DATASET.STD = [0.229, 0.224, 0.225]
_C.DATASET.ROOT = ''  # will be update later
_C.DATASET.NUM_CLASS = 10  # will be update later
_C.DATASET.IMAGE_RESIZE = (256, 256)
_C.DATASET.IMAGE_SIZE = (224, 224)
_C.DATASET.RANDOM_CROP = True
_C.DATASET.RANDOM_FLIP = True

# Model
_C.MODEL = CfgNode()
_C.MODEL.BACKBONE = 'ResNet50'
_C.MODEL.STIEFEL = True
_C.MODEL.NAME = 'DRDA'
_C.MODEL.BOTTLENECK_DIM = 256
_C.MODEL.DOWN_SAMPLE = False
_C.MODEL.LAMBDA_CLF = 1.0
_C.MODEL.LAMBDA_DIST_C = 1.0
_C.MODEL.LAMBDA_DIST = 0.5
_C.MODEL.LAMBDA_ANGULAR = 1.0
_C.MODEL.LAMBDA_WASSERSTEIN = 0.001
_C.MODEL.LAMBDA_KL = 1.0
_C.MODEL.THETA_MOVE = 0.3
_C.MODEL.SOURCE_GAMMA = 0.5
_C.MODEL.BURN_IN = 0.1

# Train
_C.TRAIN = CfgNode()
_C.TRAIN.BATCH_SIZE_PER_GPU = 32
_C.TRAIN.START_EPOCH = 0
_C.TRAIN.END_EPOCH = 100004
_C.TRAIN.CLUSTER_INC_FRQ = 5000

_C.TRAIN.OPTIM = 'SGD'
_C.TRAIN.LR = 0.001
_C.TRAIN.LR_GAMMA = 10
_C.TRAIN.BETA1 = 0.5  # Adam
_C.TRAIN.BETA2 = 0.999  # Adam
_C.TRAIN.WEIGHT_DECAY = 0.0005  # SGD
_C.TRAIN.MOMENTUM = 0.9  # SGD
_C.TRAIN.NESTEROV = True  # SGD
_C.TRAIN.SEED = 2019
_C.TRAIN.PRINT_ITER = 20
_C.TRAIN.NUM_ITER = 32
_C.TRAIN.NUM_SHOW_MAX = 16

_C.TRAIN.LOG_EMBEDDING = False

# Validation
_C.VAL = CfgNode()
_C.VAL.FRQ = 200
_C.VAL.BATCH_SIZE_PER_GPU = 32
_C.VAL.CHECKPOINT = 'model_best.pth'
_C.VAL.RESULT = './results'

_C.DEBUG = CfgNode()
_C.DEBUG.DEBUG = False

config = _C


def update_config(cfg, args):
    cfg.defrost()
    cfg.merge_from_file(args['cfg'])
    arg_list = []
    for key in args:
        if key != 'cfg':
            arg_list.append(key)
            arg_list.append(args[key])
    cfg.merge_from_list(arg_list)
    cfg.freeze()


if __name__ == '__main__':
    import sys

    with open(sys.argv[1], 'w') as fp:
        print(_C, file=fp)
