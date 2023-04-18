import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
from tqdm import tqdm


class Entropy(nn.Module):
    def __init__(self):
        super(Entropy, self).__init__()

    def forward(self, x):
        b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
        b = -1.0 * b.sum() / x.size(0)
        return b


def grl_hook(coeff):
    def fun1(grad):
        return -coeff * grad.clone()

    return fun1


def CDANLoss(category_logits, feature, cd_net, entropy=None, coeff=None, random_layer=None):
    if random_layer is None:
        op_out = torch.bmm(category_logits.unsqueeze(2), feature.unsqueeze(1))
        cd_out = cd_net(
            op_out.view(-1, category_logits.size(1) * feature.size(1)))
    else:
        rand_out = random_layer([feature, category_logits])
        cd_out = cd_net(rand_out.view(-1, rand_out.size(1)))
    batch_size = category_logits.size(0) // 2
    domain_labels = torch.ones_like(cd_out)
    domain_labels[:batch_size] *= 0

    if entropy is not None:
        entropy.register_hook(grl_hook(coeff))
        entropy = 1 + torch.exp(-entropy)
        entropy = 1.0 + torch.exp(-entropy)
        source_mask = torch.ones_like(entropy)
        source_mask[feature.size(0) // 2:] = 0
        source_weight = entropy * source_mask
        target_mask = torch.ones_like(entropy)
        target_mask[0:feature.size(0) // 2] = 0
        target_weight = entropy * target_mask
        weight = source_weight / torch.sum(source_weight).detach().item() + \
            target_weight / torch.sum(target_weight).detach().item()
        return torch.sum(weight.view(-1, 1) * nn.BCEWithLogitsLoss(reduction='none')(ad_out, dc_target)) / torch.sum(
            weight).detach().item()
    else:
        return nn.BCEWithLogitsLoss()(cd_out, domain_labels)


def pairwise_distance(features, squared=False):
    """
    Compute the pairwise distance matrix with numerical stability.
    output[i, j] = || feature[i, :] - feature[j, :] ||_2

    :param features: 2-D Tensor of size [number of data, feature dimension]
    :param squared: Boolean, whether or not to square the pairwise distance
    :return: 2-D Tensor of size [number of data, number of data]
    """
    pairwise_distances_squared = torch.add(
        torch.sum(torch.pow(features, exponent=2.0), dim=1, keepdim=True),
        torch.sum(torch.pow(features.transpose(0, 1), exponent=2.0), dim=0, keepdim=True)) - 2.0 * torch.matmul(
        features, features.transpose(0, 1))

    # Deal with numerical inaccuracies. Set small negative to zero
    pairwise_distances_squared = torch.clamp_min(
        pairwise_distances_squared, 0.0)
    # Get mask where the zero distance are at
    error_mask = pairwise_distances_squared.le(0.0)

    if squared:
        pairwise_distances = pairwise_distances_squared
    else:
        pairwise_distances = torch.sqrt(
            pairwise_distances_squared + error_mask.float() * 1e-16)

    # Undo conditionally adding 1e-16
    num_data = features.size(0)
    mask_offdiagonals = torch.ones_like(pairwise_distances) - torch.diag(
        torch.ones([num_data]).to(pairwise_distances.device))
    pairwise_distances = torch.mul(pairwise_distances, mask_offdiagonals)
    return pairwise_distances


def compute_facility_energy(pairwise_distances, centroid_ids):
    """
    Compute the average travel distance to the assigned centroid.

    :param pairwise_distances: 2-D Tensor of pairwise distances
    :param centroid_ids: 1-D Tensor of indices
    :return: facility_energy: dtypes.float32 scalar
    """
    min_val, _ = torch.min(torch.index_select(
        pairwise_distances, 0, centroid_ids), dim=0)
    return -1.0 * torch.sum(min_val)


def transformation_from_points(pts1, pts2):
    """
    Perform Procrustes Analysis for two set of points.

    ref:
        https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem

    :param pts1: NxD points list
    :type pts1: Tensor
    :param pts2: NxD points list
    :type pts2: Tensor
    :return:
    """
    # c1 = torch.mean(pts1, dim=0)
    # c2 = torch.mean(pts2, dim=0)
    N, D = pts1.size()
    c1 = pts1[0, :].clone()
    c2 = pts2[0, :].clone()
    pts1_mean_norm = pts1 - c1
    pts2_mean_norm = pts2 - c2

    s1 = torch.std(pts1_mean_norm)
    s2 = torch.std(pts2_mean_norm)
    pts1_std_norm = pts1_mean_norm / s1
    pts2_std_norm = pts2_mean_norm / s2

    for_svd_mat = torch.mm(pts2_std_norm.T, pts1_std_norm).T
    # https://github.com/pytorch/pytorch/issues/28293#issuecomment-613518799
    try:
        U, S, Vt = torch.svd(for_svd_mat)
    except:
        # torch.svd may have convergence issues for GPU and CPU.
        U, S, Vt = torch.svd(for_svd_mat + 1e-4 *
                             for_svd_mat.mean() * torch.rand_like(for_svd_mat))
    R = torch.mm(U, Vt)
    scale = s1 / s2

    padded_row = torch.zeros((1, D + 1), dtype=pts1.dtype, device=pts1.device)
    padded_row[:, -1] += 1.0
    translation = c1.unsqueeze(0).T - torch.mm(scale * R, c2.unsqueeze(0).T)
    return torch.cat([torch.cat([scale * R, translation], dim=1),
                      padded_row], dim=0)


def rotation_matrix_of(vector1, vector2):
    """
    Calculate the rotation matrix from vector1 rotate to vector2

    :param vector1: Dx1 dimension vector
    :type vector1: Tensor
    :param vector2: Dx1 dimension vector
    :type vector2: Tensor
    """
    # scale = vector2.norm() / vector1.norm()
    normalized_vector1 = vector1 / torch.norm(vector1)
    normalized_vector2 = vector2 / torch.norm(vector2)
    vector_n = normalized_vector1 + normalized_vector2
    dividend = torch.mm(vector_n, vector_n.T)
    denominator = torch.mm(vector_n.T, vector_n)
    rotate_mat = 2 * dividend / denominator - \
        torch.eye(vector_n.size(0), device=vector1.device)
    return rotate_mat


def mean_displacement_norm_vector(pts1, pts2):
    """
    Estimate radial like shape rotation alignment

    :param pts1: NxD point set, first point is centroid
    :type pts1: Tensor
    :param pts2: NxD point set, first point is centroid
    :type pts2: Tensor
    """
    num_points, num_dim = pts1.size()
    num_vec = num_points - 1

    # transpose to Dx(N-1) and make better for rotation calculation
    vectors1 = (pts1[1:, :] - pts1[0, :]).T
    vectors2 = (pts2[1:, :] - pts2[0, :]).T
    cosine_values = torch.cosine_similarity(vectors1, vectors2, dim=0)

    # normalize the original vector to unit vector
    unit_vectors1 = vectors1 / vectors1.norm(dim=0).unsqueeze(0)
    unit_vectors2 = vectors2 / vectors2.norm(dim=0).unsqueeze(0)

    orthogonal_vector = torch.zeros_like(unit_vectors1[:, 0]).unsqueeze(1)
    orthogonal_vector[0] += 1.0

    sum_residual = 0
    for i in range(num_vec):
        rot_mat = rotation_matrix_of(
            unit_vectors1[:, i].unsqueeze(1), orthogonal_vector)
        residual = torch.mm(rot_mat,
                            (unit_vectors2[:, i] / cosine_values[i] - unit_vectors1[:, i]).unsqueeze(1))
        sum_residual = sum_residual + torch.abs(residual)

    mean_residual = sum_residual / num_vec
    return mean_residual


def _compute_nmi_score(labels, predictions):
    labels_true = labels.detach().cpu().numpy()
    labels_pred = predictions.detach().cpu().numpy()
    return metrics.normalized_mutual_info_score(labels_true, labels_pred)


def compute_clustering_score(labels, predictions, margin_type):
    r"""
    Compute the clustering score via sklearn.metrics function
    There are various ways to compute the clustering score. Intuitively,
    we want to measure the agreement of two clustering assignments (labels vs
    predictions) ignoring the permutations and output a score from zero to one.
    (where the values close to one indicate significant agreement).

    This code supports following scoring functions:

    - nmi: normalized mutual information
    - ami: adjusted mutual information
    - ari: adjusted random index
    - vmeasure: v-measure
    - const: indicator checking whether the two clusterings are the same.

    .. Link: `http://scikit-learn.org/stable/modules/classes.html#clustering-metrics`

    :param labels: 1-D Tensor, ground truth cluster assignment
    :param predictions: 1-D Tensor, predicted cluster assignment
    :param margin_type: Type of structured margin to use, Default is nmi
    :return clustering_score: dtypes.float32 scalar.
            The possible valid values are from zero to one
            Zero means the worst clustering and one means the perfect clustering
    :raise
        ValueError: margin_type is not recognized

    """
    margin_type_to_func = {
        'nmi': _compute_nmi_score,
        # 'ami': _compute_ami_score,
        # 'ari': _compute_ari_score,
        # 'vmeasure': _compute_vmeasure_score,
        # 'const': _compute_zeroone_score
    }
    if margin_type not in margin_type_to_func:
        raise ValueError('Unrecognized margin_type: {}'.format(margin_type))
    clustering_score_fn = margin_type_to_func[margin_type]
    return clustering_score_fn(labels, predictions).squeeze()


def update_1d_tensor(y, index, value):
    """
    Updates 1d tensor y so that y[index] = value.

    :param y: 1-D Tensor.
    :param index: index of y to modify.
    :param value: new value to write at y[index].
    :return:
        y_mod: 1-D Tensor. Tensor y after the update.
    """
    y_before = y[:index]
    y_after = y[index + 1:]
    y_mod = torch.cat([y_before, value.view(1), y_after], dim=0)
    return y_mod


def compute_gt_cluster_score(pairwise_distances, labels):
    """
    Compute ground truth facility location score

    Loop over each unique classes and compute average travel distances.

    :param pairwise_distances: 2-D Tensor of pairwise distances.
    :param labels: 1-D Tensor of ground truth cluster assignment.
    :return: gt_cluster_score: dtypes.float32 score.
    """
    unique_class_ids = torch.unique(labels)
    num_classes = unique_class_ids.size(0)
    gt_cluster_score = 0.0

    for iteration in range(num_classes):
        mask = labels.eq(unique_class_ids[iteration])
        this_cluster_ids = torch.where(mask)[0]
        pairwise_distances_subset = torch.transpose(
            torch.index_select(
                torch.transpose(
                    torch.index_select(pairwise_distances,
                                       0, this_cluster_ids),
                    0, 1),
                0, this_cluster_ids),
            0, 1)
        this_cluster_score = -1.0 * torch.min(
            torch.sum(pairwise_distances_subset, dim=0))
        gt_cluster_score += this_cluster_score
    return gt_cluster_score


def update_medoid_per_cluster(pairwise_distances, pairwise_distances_subset,
                              labels, chosen_ids,
                              cluster_member_ids,
                              cluster_idx, margin_multiplier,
                              margin_type):
    """
    Updates the cluster medoid per cluster.

    :param pairwise_distances: 2-D Tensor of pairwise distances.
    :param pairwise_distances_subset: 2-D Tensor of pairwise distances.
    :param labels: 1-D Tensor of ground truth cluster assignment.
    :param chosen_ids: 1-D Tensor of cluster centroid indices.
    :param cluster_member_ids: 1-D Tensor of cluster member indices for one cluster.
    :param cluster_idx: Index of this one cluster.
    :param margin_multiplier: multiplication constant.
    :param margin_type: Type of structured margin to use. Default is nmi.
    :return:
        chosen_ids: Updated 1-D Tensor of cluster centroid indices.
    """
    # pairwise_distances_subset is of size [p, 1, 1, p],
    #   the intermediate dummy dimensions at [1, 2]
    #   makes this code work in the edge case where p=1
    #   this happens if the cluster size is one.
    scores_fac = -1.0 * torch.sum(pairwise_distances_subset, dim=0)
    num_candidates = cluster_member_ids.size(0)
    scores_margin = torch.zeros([num_candidates]).to(pairwise_distances.device)

    for iteration in range(num_candidates):
        candidate_medoid = cluster_member_ids[iteration]
        chosen_ids[cluster_idx] = candidate_medoid
        predictions = get_cluster_assignment(pairwise_distances, chosen_ids)
        metric_score = compute_clustering_score(
            labels, predictions, margin_type)
        scores_margin[iteration] = 1.0 - metric_score

    candidate_scores = scores_fac + margin_multiplier * scores_margin

    argmax_index = torch.argmax(candidate_scores, dim=0)

    best_modoid = cluster_member_ids[argmax_index].long()
    chosen_ids = update_1d_tensor(chosen_ids, cluster_idx, best_modoid)
    return chosen_ids


def update_all_medoids(pairwise_distances, predictions, labels, chosen_ids,
                       margin_multiplier, margin_type):
    """
    Updates all cluster medoids a cluster at a time

    :param pairwise_distances: 2-D Tensor of pairwise distance.
    :param predictions: 1-D Tensor of predicated cluster assignment.
    :param labels: 1-D Tensor of ground truth cluster assignment.
    :param chosen_ids: 1-D Tensor of cluster centroid indices.
    :param margin_multiplier: multiplication constant.
    :param margin_type: Type of structured margin to use. Default is nmi.
    :return:
        chosen_ids: Updates 1-D Tensor of cluster centroid indices
    """

    unique_class_ids = labels.unique()
    num_classes = unique_class_ids.size(0)
    for iteration in range(num_classes):
        mask = torch.eq(predictions.long(), chosen_ids[iteration])
        this_cluster_ids = torch.where(mask)[0]
        pairwise_distances_subset = torch.transpose(
            torch.index_select(
                torch.index_select(pairwise_distances, 0,
                                   this_cluster_ids).transpose(0, 1),
                0, this_cluster_ids
            ), 0, 1)

        chosen_ids = update_medoid_per_cluster(pairwise_distances, pairwise_distances_subset, labels, chosen_ids,
                                               this_cluster_ids, iteration, margin_multiplier, margin_type)
    return chosen_ids


def get_cluster_assignment(pairwise_distances, centroid_ids):
    """
    Assign data points to the nearest centroids.

    :param pairwise_distances: 2-D Tensor of pairwise distances.
    :param centroid_ids: 1-D Tensor of centroid indices.
    :return:
        y_fixed: 1-D tensor of cluster assignment.
    """
    predictions = torch.argmin(
        torch.index_select(pairwise_distances, 0, centroid_ids),
        dim=0)
    return centroid_ids[predictions]


def compute_augmented_facility_locations_pam(pairwise_distances,
                                             labels,
                                             margin_multiplier,
                                             margin_type,
                                             chosen_ids,
                                             pam_max_iter=5):
    """
    Refine the cluster centroids with PAM local search.

    For fixed iterations, alternate between updating the cluster assignment
        and updating cluster medoids.

    :param pairwise_distances: 2-D Tensor of pairwise distances.
    :param labels: 1-D Tensor of ground truth cluster assignment.
    :param margin_multiplier: multiplication constant.
    :param margin_type: Type of structured margin to use. Default is nmi.
    :param chosen_ids: 1-D Tensor of initial estimate of cluster centroids.
    :param pam_max_iter: Number of refinement iterations.
    :return:
        chosen_ids: Updated 1-D Tensor of cluster centroid indices.
    """
    for _ in range(pam_max_iter):
        # Update the cluster assignment given the chosen_ids (S_pred)
        predictions = get_cluster_assignment(pairwise_distances, chosen_ids)

        # update the medoids per each cluster
        chosen_ids = update_all_medoids(pairwise_distances, predictions, labels,
                                        chosen_ids, margin_multiplier,
                                        margin_type)
    return chosen_ids


def _find_loss_augmented_facility_idx(pairwise_distances, labels,
                                      chosen_ids, candidate_ids,
                                      margin_multiplier,
                                      margin_type):
    """
    Find the next centroid that maximizes the loss augmented inference.

    :param pairwise_distances: 2-D Tensor of pairwise distances.
    :param labels: 1-D Tensor of ground truth cluster assignment.
    :param chosen_ids: 1-D Tensor of current centroid indices.
    :param candidate_ids: 1-D Tensor of candidate indices.
    :param margin_multiplier: multiplication constant.
    :param margin_type: Type of structured margin to use. Default is nmi.
    :return:
        integer index.
    """
    num_candidates = candidate_ids.size(0)
    pairwise_distances_chosen = torch.index_select(
        pairwise_distances, 0, chosen_ids)
    pairwise_distances_candidate = torch.index_select(
        pairwise_distances, 0, candidate_ids)
    pairwise_distances_chosen_tile = pairwise_distances_chosen.repeat(
        1, num_candidates)

    min_distances, _ = torch.min(torch.cat([pairwise_distances_chosen_tile,
                                            pairwise_distances_candidate.view([1, -1])], dim=0),
                                 dim=0, keepdim=True)
    candidate_scores = -1.0 * torch.sum(
        min_distances.view(num_candidates, -1),
        dim=1
    )

    nmi_scores = torch.zeros([num_candidates]).to(pairwise_distances.device)
    for iteration in range(num_candidates):
        predictions = get_cluster_assignment(
            pairwise_distances,
            torch.cat((chosen_ids, candidate_ids[iteration].view(1)), dim=0))
        nmi_score_i = compute_clustering_score(
            labels, predictions, margin_type)
        pad_before = torch.zeros([iteration])
        pad_after = torch.zeros([num_candidates - 1 - iteration])
        nmi_scores += torch.cat([pad_before, torch.tensor(1.0 - nmi_score_i).view(1), pad_after], 0).to(
            pairwise_distances.device)

    candidate_scores = candidate_scores + margin_multiplier * nmi_scores
    argmax_index = candidate_scores.argmax(dim=0)
    return candidate_ids[argmax_index]


def compute_augmented_facility_locations(pairwise_distances, labels, all_ids,
                                         margin_multiplier, margin_type):
    """
    Compute the centroid locations

    :param pairwise_distances: 2-D Tnesor of pairwise distances.
    :param labels: 1-D Tensor of ground truth cluster assignment.
    :param all_ids: 1-D Tensor of all data
    :param margin_multiplier: utliplication constant
    :param margin_type: Type of structured margin to use. Default is nmi
    :return:
        chosen_ids: 1-D Tensor of chosen centroid indices
    """
    unique_class_ids = torch.unique(labels)
    num_classes = unique_class_ids.size(0)
    chosen_ids = torch.zeros([0], dtype=torch.long).to(
        pairwise_distances.device)

    for iteration in tqdm(range(num_classes), ncols=80, desc='Facility'):
        candidate_ids = relative_complement(
            all_ids, chosen_ids).to(pairwise_distances.device)
        new_chosen_idx = _find_loss_augmented_facility_idx(pairwise_distances,
                                                           labels, chosen_ids,
                                                           candidate_ids,
                                                           margin_multiplier,
                                                           margin_type)
        chosen_ids = torch.cat([chosen_ids, new_chosen_idx.view(1)], dim=0)
    return chosen_ids


def cluster_loss(labels,
                 embeddings,
                 margin_multiplier,
                 enable_pam_finetuning=True,
                 margin_type='nmi',
                 print_losses=False):
    r"""
    Compute the clustering loss.

    :param labels: 2-D Tensor of labels of shape [batch_size, 1]
    :param embeddings: 2-D Tensor of embeddings of shape
        [batch_size, embedding_dimension]. Embeddings should be L2 normalized
    :param margin_multiplier: float32 scalar, multiplier on the structured margin term
    :param enable_pam_finetuning: Boolean, whether to run local pam refinement
    :param margin_type: Type of structured margin to use
    :param print_losses: Boolean
    :return
        clustering_loss: A float32 scalar 'Tensor'
    :raise
        ImportError: If sklearn dependency is not installed

    """
    pairwise_distances = pairwise_distance(embeddings)
    all_ids = torch.arange(embeddings.size(0))
    # Compute the loss augmented inference and get the cluster centroids
    # That is from the algorithm 1
    chosen_ids = compute_augmented_facility_locations(pairwise_distances, labels,
                                                      all_ids, margin_multiplier,
                                                      margin_type)
    # Given the predicted centroids, compute the clustering score.
    score_pred = compute_facility_energy(pairwise_distances, chosen_ids)

    # branch whether to use PAM finetuning.
    if enable_pam_finetuning:
        chosen_ids = compute_augmented_facility_locations_pam(pairwise_distances,
                                                              labels,
                                                              margin_multiplier,
                                                              margin_type,
                                                              chosen_ids)
        score_pred = compute_facility_energy(pairwise_distances, chosen_ids)

    # Given the predicted centroids, compute the cluster assigments.
    predictions = get_cluster_assignment(pairwise_distances, chosen_ids)

    # Compute the clustering (i.e. NMI) score between the two assignments.
    clustering_score_pred = compute_clustering_score(
        labels, predictions, margin_type)

    # Compute the clustering score from labels.
    score_gt = compute_gt_cluster_score(pairwise_distances, labels)

    # Compute the hinge loss.
    clustering_loss = torch.clamp_min(
        score_pred + margin_multiplier *
        (1.0 - clustering_score_pred) - score_gt,
        0.0)

    if print_losses:
        print(clustering_loss)

    return clustering_loss


class FacilityLoss(nn.Module):
    def __init__(self):
        super(FacilityLoss, self).__init__()
        self._S_set = []


def relative_complement(universal_set, subset):
    device = universal_set.device
    np_universal_set = universal_set.detach().cpu().numpy()
    np_subset = subset.detach().cpu().numpy()
    new_set = np.setdiff1d(np_universal_set, np_subset)
    return torch.tensor(new_set)


def sort_rows(matrix, num_rows):
    matrix_T = matrix.T
    sorted_matrix_T = torch.topk(matrix_T, num_rows)[0]
    return sorted_matrix_T.T


def sliced_wasserstein(xs, xt, num_projection=128, random_projection=None):
    # TODO: create version to support the source and target with different number of points.
    random_projection_dim = xs.size(1)
    if random_projection is None:
        random_projection = torch.rand(random_projection_dim, num_projection)
    projection = random_projection / torch.sum(random_projection ** 2.0,
                                               dim=0, keepdim=True)

    xs_p = torch.matmul(xs, projection)
    xt_p = torch.matmul(xt, projection)

    xs_p = sort_rows(xs_p, xs.size(0))
    xt_p = sort_rows(xt_p, xt.size(0))
    wdist = torch.mean(torch.square(xs_p - xt_p))
    return wdist
