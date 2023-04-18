import torch
import torch.nn.functional as F


def accuracy(pred, labels):
    r"""
    Computes the accuracy

    :param pred:
    :param labels:
    :return:
    """
    assert pred.size(0) == labels.size(0)
    with torch.no_grad():
        batch_size = pred.size(0)
        _, pred_label = torch.max(pred, 1)
        correct_cnt = (pred_label == labels.data).sum()
    return correct_cnt.type(torch.float) / batch_size, batch_size


def topk_accuracy(output, target, topk=(1,)):
    r"""
    Computes the accuracy over the k top predictions for the specified values of k

    :param output: NxC prediction output where N indicates to number of samples,
        C indicates number of category
    :param target: N
    :param topk:
    :return:
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def pairwise_cos(pred1, pred2):
    """
    Compare the element wise cosine similarity

    :param pred1:
    :param pred2:
    :return:
    """

    batch_size = pred1.size(0)
    cos_matrix = torch.empty((batch_size, batch_size), dtype=torch.float64)
    for i in range(batch_size):
        cos_matrix[i, :] = torch.cosine_similarity(pred1[i, :].unsqueeze(0), pred2)
    return cos_matrix.mean()


def pairwise_kl(pred1, pred2):
    r"""
    compute the kl divergence

    :param pred1: n-by-c matrix, where c is the dimension of prediction
    :param pred2: n-bu-c matrix, where c is the dimension of prediction
    :return kl_mean:
    """
    assert pred1.size(0) == pred2.size(0)
    batch_size = pred1.size(0)
    kl_matrix = torch.empty(batch_size, batch_size, dtype=torch.float64)
    # for i in range(batch_size):
    #     for j in range(batch_size):
    #         kl_matrix[i, j] = F.kl_div(pred1[i], pred2[j])
    # return torch.mean(kl_matrix)
    return F.kl_div(pred1, pred2)
