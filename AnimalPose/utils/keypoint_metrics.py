import numpy as np

PCK_THRESH = [0.01, 0.025, 0.05, 0.1, 0.125, 0.15, 0.175, 0.2, 0.25, 0.5]


def kp_confidence_mask(confs, thresh):
    masks = []
    inv_masks = []
    for c in confs:
        m = c > thresh
        im = c < thresh
        masks += [m]
        inv_masks += [im]

    mask = masks[0]
    for m in masks[1:]:
        mask = np.logical_and(mask, m)

    inv_mask = inv_masks[0]
    for m in inv_masks[1:]:
        inv_mask = np.logical_and(inv_mask, m)

    only_one_detection = masks[0]
    for m in masks[1:]:
        only_one_detection = np.logical_xor(only_one_detection, m)

    both_same = np.logical_not(only_one_detection)

    return mask, only_one_detection, both_same, inv_mask


def l2_metric(kps1, kps2,
              confs1=None, confs2=None,
              c_thresh=0.8,
              fixed_penalty=0.531):
    '''Calculates the mean pointwise L2 distance of sets of keypoints.

    Args:
        kps1, kps2 (np.array): Kps of shape ``[(*), K, 2]``.
        confs1, confs2 (np.array): confidences for both sets of keypoints. If
            both are given the keypoints are filtered to only consist of pairs
            of keypoints with ``conf > c_thresh``.
        c_thresh (float): See above.
        fixed_penalty (float): Fixed distances added as penalty for keypoints
            detected even though they should not and vice versa.

    Returns:
        float: Mean pointwise l2 distance.
        float: Standart deviation of those distances.
    '''

    dists = np.linalg.norm(kps1 - kps2, axis=-1)

    if confs1 is not None and confs2 is not None:
        both_mask, only_one_mask, _, _ = kp_confidence_mask([confs1, confs2],
                                                            c_thresh)

        dists[only_one_mask] = fixed_penalty

        dists = dists[np.logical_and(only_one_mask, both_mask)]

    return {'mean': np.mean(dists), 'std': np.std(dists)}


def pck_metric(kps1, kps2, confs1=None, confs2=None, threshold=0.1,
               c_thresh=0.8, no_mean=False):
    '''Calculates the percentage of keypoint pairs within the threshold.

    Args:
        kps1, kps2 (np.array): Kps of shape ``[(*), K, 2]``.
        threshold (float): Maximum distance at which a pair of keypoints is
            considered to be correctly matched.
        confs1, confs2 (np.array): confidences for both sets of keypoints. If
            both are given the keypoints are filtered to only consist of pairs
            of keypoints with ``conf > c_thresh``.
        c_thresh (float): See above.
        no_mean (bool): Return full correctness array with bool values for all
            examples and keypoints.

    Returns:
        if ``no_mean == False``:
            float: Percentage of correct keypoints.
            np.ndarray: Percentage of correct detections per keypoint.
        else:
            np.ndarray: Correct dection array of type bool with shape
                ``[(*), K, 2]``
    '''

    dists = np.linalg.norm(kps1 - kps2, axis=-1)

    correct = dists <= threshold

    if confs1 is not None and confs2 is not None:
        _, _, same_m, inv_m = kp_confidence_mask([confs1, confs2], c_thresh)

        # Wrong detection if not both detected with confidence > c_thres
        correct[np.logical_not(same_m)] = False

        # Correct detection if both detected with confidence <= c_thresh
        correct[inv_m] = True

    if not no_mean:
        return {'mean': np.mean(correct), 'per_kp': np.mean(correct, axis=0)}
    else:
        return {'correct_map': correct}


def multi_thresh_pck(kps1, kps2, confs1=None, confs2=None, thresholds=[],
        c_thresh=0.8, no_mean=False):
    '''Same as :function:`pck_metric`, but evaluates it for a list of
    thresholds.'''

    return {t: pck_metric(kps1, kps2, confs1, confs2, t, c_thresh, no_mean)
            for t in thresholds}