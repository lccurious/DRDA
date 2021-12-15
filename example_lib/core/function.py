import torch
import torch.nn as nn
from torch.nn import functional as F


def conv_init(m):
    """Initialize the convolution kernel

    :param m: Conv module
    :type m: nn.Module
    """
    nn.init.kaiming_normal_(m.weight, mode='fan_out')
    if m.bias is not None:
        nn.init.constant_(m.bias, 0)


def bn_init(bn, scale):
    """Initialize the batch norm kernel

    :param bn: Batchnorm module
    :type bn: nn.Module
    :param scale: The normalize scale
    :type scale: float
    """
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)
