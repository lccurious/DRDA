import torch
import torch.nn as nn
from torch import autograd

from core.function import init_weights
from models.backbones import StiefelLinear


class LetNet5(nn.Module):
    def __init__(self, cfg):
        super(LetNet5, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=(5, 5)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        )

        self.conv2_1 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=(5, 5)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        )

        self.conv2_2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=(5, 5)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 120, kernel_size=(2, 2)),
            nn.ReLU()
        )

        self.fc1 = nn.Sequential(
            nn.Linear(120, 84),
            nn.ReLU()
        )

        self.fc2 = nn.Sequential(
            nn.Linear(84, 10),
            nn.LogSoftmax(dim=-1)
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv3(self.conv2_2(out) + self.conv2_1(out))
        out = out.view(x.size(0), -1)
        y = self.fc1(out)
        y = self.fc2(y)
        return out, y


class DTNBackbone(nn.Module):
    def __init__(self):
        super(DTNBackbone, self).__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.Dropout2d(0.1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(128),
            nn.Dropout2d(0.3),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(256),
            nn.Dropout2d(0.5),
            nn.ReLU()
        )
        self._in_features = 256 * 4 * 4

    def output_dim(self):
        return self._in_features

    def forward(self, x):
        x = self.conv_layer(x)
        x = x.view(x.size(0), -1)
        return x

    def get_parameters(self):
        param_list = [
            {'params': self.parameters(), 'lr_mult': 1.0, 'decay_mult': 2.0}
        ]
        return param_list


class LeNetBackbone(nn.Module):
    def __init__(self):
        super(LeNetBackbone, self).__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(1, 20, kernel_size=5),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(20, 50, kernel_size=5),
            nn.Dropout2d(0.5),
            nn.MaxPool2d(2),
            nn.ReLU()
        )
        self._in_features = 50 * 4 * 4

    def output_dim(self):
        return self._in_features

    def forward(self, x):
        x = self.conv_layer(x)
        x = x.view(x.size(0), -1)
        return x

    def get_parameters(self):
        param_list = [
            {'params': self.parameters(), 'lr_mult': 1.0, 'decay_mult': 2.0}
        ]
        return param_list


class FeatureBottleneck(nn.Module):
    def __init__(self, feature_dim, bottleneck_dim=256, type='origin'):
        super(FeatureBottleneck, self).__init__()
        self.bn = nn.BatchNorm1d(bottleneck_dim, affine=True)
        self.dropout = nn.Dropout(0.5)
        self.bottleneck = nn.Linear(feature_dim, bottleneck_dim)
        self.bottleneck.apply(init_weights)
        self.type = type

    def forward(self, x):
        x = self.bottleneck(x)
        if self.type == 'bn':
            x = self.bn(x)
            x = self.dropout(x)
        return x

    def get_parameters(self):
        param_list = [
            {'params': self.parameters(), 'lr_mult': 1.0, 'decay_mult': 2.0}
        ]
        return param_list


class SoftClusterAssignment(nn.Module):
    r"""
    Trainable clustering centroids which compute the soft assignment probability according to:

    .. math::
        q_{ij}=\frac{(1+d(\mathbf{z}_{i}-\mathbf{a}, \mathbf{v}_{j})/\alpha)^{-\frac{\alpha+1}{2}}}{\sum_{j'}(1+d(\mathbf{z}_{i}-\mathbf{a}, \mathbf{v}_{j'})/\alpha)^{-\frac{\alpha+1}{2}}}

    """

    def __init__(self, num_classes, bottleneck_dim=256, alpha=1, similarity_type='cosine'):
        super(SoftClusterAssignment, self).__init__()
        self.alpha = alpha
        self.similarity_type = similarity_type
        self.register_buffer('eps', torch.tensor(1e-8, requires_grad=False))
        if self.similarity_type == 'cosine':
            self.fc = nn.Linear(bottleneck_dim, num_classes, bias=False)
        self.fc.apply(init_weights)

    def forward(self, x):
        if self.similarity_type == 'cosine':
            sim = self.fc(x) / torch.max(x.norm(dim=1).unsqueeze(1) * self.fc.weight.norm(dim=1).unsqueeze(0), self.eps)
        else:
            raise NotImplemented
        prob = torch.pow(1 + sim / self.alpha, (self.alpha + 1) / 2)
        return nn.Softmax(dim=1)(prob)

    def get_parameters(self):
        param_list = [
            {'params': self.parameters(), 'lr_mult': 1.0, 'decay_mult': 2.0}
        ]
        return param_list

    def proxy_vectors(self):
        return self.fc.weight


class FeatureClassifier(nn.Module):
    def __init__(self, num_classes, bottleneck_dim=256, type='linear'):
        super(FeatureClassifier, self).__init__()
        if type == 'linear':
            self.fc = nn.Linear(bottleneck_dim, num_classes)

        self.fc.apply(init_weights)

    def forward(self, x):
        x = self.fc(x)
        return x

    def get_parameters(self):
        param_list = [
            {'params': self.parameters(), 'lr_mult': 1.0, 'decay_mult': 2.0}
        ]
        return param_list


class KantorovichPotential(nn.Module):
    def __init__(self, in_dims):
        super(KantorovichPotential, self).__init__()
        self.map_module = nn.Sequential(
            nn.Linear(in_dims, in_dims*2),
            nn.BatchNorm1d(num_features=in_dims*2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(in_dims*2, in_dims*4),
            nn.BatchNorm1d(num_features=in_dims*4),
            nn.LeakyReLU(),
            nn.Linear(in_dims*4, 1)
        )

    def forward(self, x):
        return self.map_module(x)

    def gradient_penalty(self, a, b):
        alpha = torch.rand(a.size(0), 1)
        alpha = alpha.expand(a.size())
        alpha = alpha.cuda()

        interpolates = alpha * a + (1-alpha) * b

        interpolates = autograd.Variable(interpolates, requires_grad=True)
        disc_interpolates = self.map_module(interpolates)

        gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                  grad_outputs=torch.ones(disc_interpolates.size()),
                                  create_graph=True, retain_graph=True, only_inputs=True)[0]
        gradients_penalty = (gradients.norm(2, dim=1) - 1) ** 2
        return gradients_penalty.mean()

    def get_parameters(self):
        param_list = [
            {'params': self.parameters(), 'lr_mult': 1.0, 'decay_mult': 2.0}
        ]
        return param_list


class LeNetStiefel(LetNet5):
    def __init__(self, cfg):
        super(LeNetStiefel, self).__init__(cfg)
        self.stiefel_linear = StiefelLinear(120, 120)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv3(self.conv2_1(out) + self.conv2_2(out))
        out = out.view(x.size(0), -1)
        out = self.stiefel_linear(out)
        y = self.fc1(out)
        y = self.fc2(y)
        return out, y

    def get_parameters(self):
        param_list = []
        for m in [self.conv1, self.conv2_1, self.conv2_2, self.conv3, self.fc1, self.fc2]:
            param_list.append(
                {'params': m.parameters(), 'lr_mult': 1.0, 'decay_mult': 2.0}
            )
        return param_list

    def get_riemann_parameters(self):
        param_list = [
            {'param': self.stiefel_linear.parameters(), 'lr_mult': 1.0, 'decay_mult': 2.0}
        ]
        return param_list
