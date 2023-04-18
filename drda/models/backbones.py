import torch
import torch.nn as nn
import numpy as np
import torchvision
import geoopt
import torch.nn.functional as F
from scipy.special import binom

from core.function import init_weights


class StiefelLinear(nn.Module):
    def __init__(self, in_channels, out_channels, bias=False):
        super(StiefelLinear, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = geoopt.ManifoldParameter(
            data=torch.Tensor(out_channels, in_channels),
            manifold=geoopt.Stiefel()
        )
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_buffer('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        self.stiefel_uniform_(self.weight)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / np.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)

    @staticmethod
    def stiefel_uniform_(tensor):
        with torch.no_grad():
            tensor.data = torch.eye(tensor.size(0), tensor.size(1))
            return tensor

    def get_parameters(self):
        param_list = [{
            'params': self.parameters(), 'lr_mult': 10, 'decay_mult': 2
        }]
        return param_list


class LSoftmaxLinear(nn.Module):
    def __init__(self, input_features, output_features, margin, device):
        super().__init__()
        self.input_dim = input_features  # number of input feature i.e. output of the last fc layer
        self.output_dim = output_features  # number of output = class numbers
        self.margin = margin  # m
        self.beta = 100
        self.beta_min = 0
        self.scale = 0.99

        self.device = device  # gpu or cpu

        # Initialize L-Softmax parameters
        self.weight = nn.Parameter(torch.FloatTensor(input_features, output_features))
        self.divisor = np.pi / self.margin  # pi/m
        self.C_m_2n = torch.Tensor(binom(margin, range(0, margin + 1, 2))).to(device)  # C_m{2n}
        self.cos_powers = torch.Tensor(range(self.margin, -1, -2)).to(device)  # m - 2n
        self.sin2_powers = torch.Tensor(range(len(self.cos_powers))).to(device)  # n
        self.signs = torch.ones(margin // 2 + 1).to(device)  # 1, -1, 1, -1, ...
        self.signs[1::2] = -1

    def calculate_cos_m_theta(self, cos_theta):
        sin2_theta = 1 - cos_theta ** 2
        cos_terms = cos_theta.unsqueeze(1) ** self.cos_powers.unsqueeze(0)  # cos^{m - 2n}
        sin2_terms = (sin2_theta.unsqueeze(1)  # sin2^{n}
                      ** self.sin2_powers.unsqueeze(0))

        cos_m_theta = (self.signs.unsqueeze(0) *  # -1^{n} * C_m{2n} * cos^{m - 2n} * sin2^{n}
                       self.C_m_2n.unsqueeze(0) *
                       cos_terms *
                       sin2_terms).sum(1)  # summation of all terms

        return cos_m_theta

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.weight.data.t())

    def find_k(self, cos):
        # to account for acos numerical errors
        eps = 1e-7
        cos = torch.clamp(cos, -1 + eps, 1 - eps)
        acos = cos.acos()
        k = (acos / self.divisor).floor().detach()
        return k

    def forward(self, input, is_tgt=False, target=None):
        if self.training and not is_tgt:
            assert target is not None
            x, w = input, self.weight
            beta = max(self.beta, self.beta_min)
            logit = x.mm(w)
            indexes = range(logit.size(0))
            logit_target = logit[indexes, target]

            # cos(theta) = w * x / ||w||*||x||
            w_target_norm = w[:, target].norm(p=2, dim=0)
            x_norm = x.norm(p=2, dim=1)
            cos_theta_target = logit_target / (w_target_norm * x_norm + 1e-10)

            # equation 7
            cos_m_theta_target = self.calculate_cos_m_theta(cos_theta_target)

            # find k in equation 6
            k = self.find_k(cos_theta_target)

            # f_y_i
            logit_target_updated = (w_target_norm *
                                    x_norm *
                                    (((-1) ** k * cos_m_theta_target) - 2 * k))
            logit_target_updated_beta = (logit_target_updated + beta * logit[indexes, target]) / (1 + beta)

            logit[indexes, target] = logit_target_updated_beta
            self.beta *= self.scale
            return logit
        else:
            assert target is None
            return input.mm(self.weight)


class ResNetFc(nn.Module):
    def __init__(self, resnet_name, use_bottleneck=True, bottleneck_dim=256, new_cls=False, num_class=1000):
        """
        No fc implemented, for output extractor features only
        """
        super(ResNetFc, self).__init__()
        resnet_dict = {
            'ResNet50': torchvision.models.resnet50,
            'ResNet101': torchvision.models.resnet101,
            'ResNet152': torchvision.models.resnet152
        }
        model_resnet = resnet_dict[resnet_name](pretrained=True)
        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4
        self.avgpool = model_resnet.avgpool
        self.feature_layers = nn.Sequential(
            self.conv1, self.bn1, self.relu, self.maxpool,
            self.layer1, self.layer2, self.layer3, self.layer4,
            self.avgpool
        )
        self.use_bottleneck = use_bottleneck
        self.new_cls = new_cls
        if new_cls:
            if self.use_bottleneck:
                self.bottleneck = nn.Linear(
                    model_resnet.fc.in_features, bottleneck_dim)
                self.fc = nn.Linear(bottleneck_dim, num_class)
                self.bottleneck.apply(init_weights)
                self.fc.apply(init_weights)
                self.__in_features = bottleneck_dim
            else:
                self.fc = nn.Linear(model_resnet.fc.in_features, num_class)
                self.fc.apply(init_weights)
                self.__in_features = model_resnet.fc.in_features
        else:
            self.fc = model_resnet.fc
            self.__in_features = model_resnet.fc.in_features

    def forward(self, x):
        """
        If the

        :param x:
        :return:
        """
        x = self.feature_layers(x)
        x = x.view(x.size(0), -1)
        if self.use_bottleneck and self.new_cls:
            x = self.bottleneck(x)
        y = self.fc(x)
        return x, y

    def output_dim(self):
        return self.__in_features

    def get_parameters(self):
        if self.new_cls:
            if self.use_bottleneck:
                param_list = [
                    {'params': self.feature_layers.parameters(), 'lr_mult': 1,
                     'decay_mult': 2},
                    {'params': self.bottleneck.parameters(), 'lr_mult': 10,
                     'decay_mult': 2},
                ]
            else:
                param_list = [
                    {'params': self.feature_layers.parameters(), 'lr_mult': 1,
                     'decay_mult': 2},
                ]
        else:
            param_list = [
                {'params': self.parameters(), 'lr_mult': 1, 'decay_mult': 2}]
        return param_list


class ResNetFcStiefelBranch(ResNetFc):
    def __init__(self, resnet_name, use_stiefel=True, use_bottleneck=True,
                 bottleneck_dim=256, new_cls=False, num_class=1000):
        super(ResNetFcStiefelBranch, self).__init__(resnet_name, use_bottleneck, bottleneck_dim, new_cls, num_class)
        self.use_stiefel = use_stiefel
        if use_stiefel:
            self.stiefel_linear = StiefelLinear(bottleneck_dim, bottleneck_dim)

    def forward(self, x, is_tgt=False, centroid=None, x_is_feature=False):
        """
        Overwrite the forward function with

        :param x: input data
        :type x: torch.Tensor
        :param is_tgt: if the input data is from target
        :type is_tgt: bool
        :param centroid: if input the centroid of data
        :param x_is_feature: if the input data is feature
        """
        if not x_is_feature:
            x = self.feature_layers(x)
            x = x.view(x.size(0), -1)
            if self.use_bottleneck and self.new_cls:
                x = self.bottleneck(x)
            if is_tgt and self.use_stiefel:
                x = self.stiefel_linear(x)
            if centroid is not None:
                x = x - centroid
        y = self.fc(x)
        return x, y

    def get_riemann_parameters(self):
        param_list = [
            {'params': self.stiefel_linear.parameters(), 'lr_mult': 10, 'decay_mult': 2}
        ]
        return param_list
