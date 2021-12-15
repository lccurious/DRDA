from yacs.config import CfgNode

_C = CfgNode()

# BASIC
_C.OUTPUT_DIR = 'outputs'
_C.LOG_DIR = 'logs'
_C.WORKERS = 4
_C.SAVE_CHECKPOINT = False
_C.AUTO_RESUME = False
_C.PIN_MEMORY = True
_C.RANK = 0
_C.USE_APEX = False

# CUDNN
_C.CUDNN = CfgNode()
_C.CUDNN.ENABLED = True
_C.CUDNN.DETERMINISTIC = True
_C.CUDNN.BENCHMARK = True


# DATASET
_C.DATASET = CfgNode()
_C.DATASET.DATASET = 'template_dataset'
_C.DATASET.ROOT = './data/template_dataset'

# MODEL
_C.MODEL = CfgNode()
_C.MODEL.NAME = 'model_name'

# TRAIN
_C.TRAIN = CfgNode()
_C.TRAIN.SEED = 2021
_C.TRAIN.BATCH_SIZE_PER_GPU = 32
_C.TRAIN.BEGIN_EPOCH = 0
_C.TRAIN.END_EPOCH = 20000
_C.TRAIN.OPTIM = 'SGD'
_C.TRAIN.LR = 0.001
_C.TRAIN.WEIGHT_DECAY = 0.001
_C.TRAIN.MOMENTUM = 0.9
_C.TRAIN.NESTEROV = True
_C.TRAIN.PRINT_ITER = 20
_C.TRAIN.BETA1 = 0.9
_C.TRAIN.BETA2 = 0.99

# VAL
_C.VAL = CfgNode()
_C.VAL.BATCH_SIZE_PER_GPU = 32

# DEBUG
_C.DEBUG = CfgNode()
_C.DEBUG.DEBUG = False

config = _C


def update_config(cfg, args):
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
        print(config, file=fp)
