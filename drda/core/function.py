import torch
import torch.nn as nn
import numpy as np


def init_weights(m):
    """
    Init the network layer respect to standard normal distribution

    :param m: module to initialize weights
    :type m: Module
    :return:
    """
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_normal_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def da_coeff(iter_num, high=1.0, low=0.0, alpha=10.0, max_iter=20000.0):
    """
    Domain Adversarial weight calculation.

    :param iter_num: iteration number of mini batch
    :type iter_num: int
    :param high: highest value for current weight
    :type high: float
    :param low: lowest value for current weight
    :type low: float
    :param alpha: control the curve raise moment and curve.
    :type alpha: float
    :param max_iter: total iteration numbers
    :type max_iter: int
    :return coefficient: the weight of domain adversarial weight at current iteration
    :rtype coefficient: float
    """
    return np.float(2.0 * (high - low) / (1.0 + np.exp(-alpha * iter_num / max_iter)) - (high - low) + low)


def mean_by_label(samples, labels, num_classes):
    """
    Select mean(samples), count() from samples group by labels order by labels asc

    :param samples: NxM samples Tensor which N is number of samples and M is number of feature dimension
    :type samples: torch.Tensor
    :param labels: Nx1 labels Tensor which N is number of samples and L is number of categories
    :type labels: torch.Tensor
    :param num_classes: The complete number of categories
    :type num_classes: int
    :return mean: the euclid center of each category of samples, if one category disappear make it zero
    :rtype mean: torch.Tensor
    """
    weight = torch.zeros(num_classes, samples.shape[0], dtype=samples.dtype).to(samples.device)  # L, N
    weight[labels, torch.arange(samples.shape[0])] = 1
    # label_count = weight.sum(dim=1)
    weight = torch.nn.functional.normalize(weight, p=1, dim=1)  # l1 normalization
    mean = torch.mm(weight, samples)  # L, F
    return mean


def batch_feature_covariance(features, labels, mean=None):
    """
    Estimate covariance of given data in feature dimension

    :param features: NxD feature
    :type features: Tensor
    :param labels: N labels
    :type labels: LongTensor
    :param mean: CxD centroids where C indicates the number of categories
    :type mean: Tensor
    :return cov_mat: CxDxD feature covariance matrix
    :rtype cov_mat: Tensor
    """
    num_class, num_dim = mean.size()
    fact = 1.0 / (num_dim - 1)
    mat = features - mean[labels, :]
    mat_transpose = mat.t()
    cov_mat = torch.zeros((num_class, num_dim, num_dim), dtype=mean.dtype, device=mean.device)
    for c in labels.unique():
        idx = labels.eq(c)
        cov_mat[c] = fact * mat_transpose[:, idx].matmul(mat[idx, :])
    return cov_mat


def soft_clustering_assignment(x, c, alpha=1.0, distance_type='euclidean'):
    """
    Compute the soft assignment from x to clusters

    :param x: NxD Tensor
    :type x: Tensor
    :param c: CxD where C indicates the number of clusters
    :type c: Tensor
    :param alpha:
    :type alpha: float
    :param distance_type: the distance type between any pair of points
    :type distance_type: str
    :return soft_labels: the [1,0]^{NxC} labels for soft assignment probabilities
    """
    # NxC
    if distance_type == 'euclidean':
        distance_matrix = pairwise_distance(x, c)
    else:
        raise NotImplemented
    logits = torch.pow(1.0 + distance_matrix / alpha, -(alpha + 1) / 2.0)
    return logits / logits.sum(dim=1, keepdim=True)


def euclidean_distances(x, y=None, squared=True):
    r"""
    Compute euclidean distance between multiple comparison of x and y, if y is None
    compute the multiple comparison of x and x,

    .. math::
        \text{dist}[i, j] = ||x[i, :] - y[j, :]||^2

    :param x: NxD Tensor
    :type x: Tensor
    :param y: NxD Tensor
    :type y: Tensor
    :param squared: if return the squared result
    :type squared: bool
    :return pairwise_distance: pairwise distance matrix
    :rtype pairwise_distance: Tensor
    """
    x_norm = (x ** 2).sum(1).view(-1, 1)
    if y is not None:
        y_t = torch.transpose(y, 0, 1)
        y_norm = (y ** 2).sum(1).view(1, -1)
    else:
        y_t = torch.transpose(x, 0, 1)
        y_norm = x_norm.view(1, -1)

    pairwise_distances_squared = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
    # Deal with numerical inaccuracies. Set small negative to zero
    pairwise_distances_squared = torch.clamp_min(pairwise_distances_squared, 0.0)
    # Get mask where the zero distance are at
    error_mask = pairwise_distances_squared.le(0.0)

    # Ensure diagonal is zero if x=y
    # if y is None:
    #     pairwise_distances_squared = pairwise_distances_squared - torch.diag(pairwise_distances_squared.diag)

    if squared:
        pairwise_distances = pairwise_distances_squared
    else:
        pairwise_distances = torch.sqrt(pairwise_distances_squared + error_mask.float() * 1e-16)

    return pairwise_distances


def pairwise_distance(x: torch.Tensor, y: torch.Tensor, device=torch.device('cuda')):
    r"""
    Calc the euclidean distance matrix with :math:`\text{dis}[i, j] = ||x[i, :] - y[j, :]||`

    :param x: NxD data samples
    :param y: MxD data samples
    :param device:
    :return:
    """
    dis = (x.unsqueeze(dim=1) - y.unsqueeze(dim=0)) ** 2.0
    dis = dis.sum(dim=-1).squeeze()
    return dis


def pairwise_cosine(x: torch.Tensor, y: torch.Tensor, device=torch.device('cuda')):
    r"""
    Calc the cosine distance matrix with :math:`\text{dis}[i, j] = \cos(x[i, :], y[j, :])`

    :param x: NxD data samples
    :param y: MxD data samples
    :param device:
    :return:
    """
    a = x.unsqueeze(dim=1)
    b = y.unsqueeze(dim=0)
    x_normalized = a / a.norm(dim=-1, keepdim=True)
    y_normalized = b / b.norm(dim=-1, keepdim=True)

    cosine = x_normalized * y_normalized
    dis_cosine = 1 - cosine.sum(dim=-1).squeeze()
    return dis_cosine


class AverageMeter(object):
    """
    Computes and stores the average and current value

    """

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0


class AverageCentroids(object):
    def __init__(self, num_centers, num_dim):
        self.val = torch.zeros(num_centers, num_dim)
        self.avg = torch.zeros(num_centers, num_dim)
        self.sum = torch.zeros(num_centers, num_dim)
        self.count = torch.zeros(num_centers)
        self.reset()

    def cuda(self):
        self.val = self.val.cuda()
        self.avg = self.avg.cuda()
        self.sum = self.sum.cuda()
        self.count = self.count.cuda()
        return self

    def reset(self):
        torch.zero_(self.val)
        torch.zero_(self.avg)
        torch.zero_(self.sum)
        torch.zero_(self.count)

    def update(self, features_sum, n):
        self.sum = features_sum
        nonzero_n_ids = n > 0
        self.val[nonzero_n_ids] = features_sum[nonzero_n_ids, :] / n[nonzero_n_ids].unsqueeze(1)
        percent = torch.zeros_like(n)
        percent[nonzero_n_ids] = n[nonzero_n_ids] / (self.count[nonzero_n_ids] + n[nonzero_n_ids])
        self.avg = self.avg * (1 - percent.unsqueeze(1)) + self.val * percent.unsqueeze(1)
        self.count += n
