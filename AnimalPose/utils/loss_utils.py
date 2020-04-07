import numpy as np
import torch
import torch.nn

from AnimalPose.utils.tensor_utils import sure_to_torch


class JointsMSELoss(torch.nn.Module):
    def __init__(self, use_target_weight=True):
        super(JointsMSELoss, self).__init__()
        self.criterion = torch.nn.MSELoss(reduction='mean')
        self.use_target_weight = use_target_weight

    def forward(self, output, target, target_weight):
        batch_size = output.size(0)
        num_joints = output.size(1)
        heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)
        loss = 0

        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze()
            heatmap_gt = heatmaps_gt[idx].squeeze()
            if self.use_target_weight:
                loss += 0.5 * self.criterion(
                    heatmap_pred.mul(target_weight[:, idx]),
                    heatmap_gt.mul(target_weight[:, idx])
                )
            else:
                loss += 0.5 * self.criterion(heatmap_pred, heatmap_gt)

        return loss / num_joints


def heatmap_loss(targets, predictions):
    crit = torch.nn.MSELoss()
    return crit(torch.from_numpy(targets), predictions)


class MSELossInstances(torch.nn.MSELoss):
    """MSELoss, which reduces to instances of the batch
    """

    def __init__(self):
        super().__init__(reduction="none")

    def forward(self, image: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        super_result = super().forward(image, target)
        reduced = super_result.mean(axis=(1, 2, 3))
        return reduced


class L1LossInstances(torch.nn.L1Loss):
    """L1Loss, which reduces to instances of the batch
    """

    def __init__(self):
        super().__init__(reduction="none")

    def forward(self, image: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        super_result = super().forward(image, target)
        reduced = super_result.mean(axis=(1, 2, 3))
        return reduced


def percentage_correct_keypoints(keypoints: np.array,
                                 predictions: np.array,
                                 thresh: float = 0.5,
                                 pck_type: str = "object",
                                 image_size: float = None):
    """

    Args:
        keypoints: Keypoints with shape [B, N, 2] or [N,2]
        predictions: Predicted keypoints with shape [B, N, 2] or [N,2]
        image_size (optional): indicates the size of the image, necessary when pck_type == "image"
        thresh: threshold for pck
        pck_type (optional): default object, indicates which way to compute the pck, e.g. via image size
                            or max object distance
                            "object": take the max of the object * alpha
                            "image": take the image width/height * alpha

    Returns: pck mean, pck per joint

    """
    if pck_type == "image" and image_size == None:
        raise ValueError(f"When using pck_type='image', then you need to pass the image_size!")
    assert pck_type in ["image", "object"], f"Got wrong pck_type, got {pck_type}"
    assert len(keypoints.shape) == 3, f"Only implemented for a batch got shape of keypoints: {keypoints.shape}"
    keypoints = sure_to_torch(keypoints).cpu()
    predictions = sure_to_torch(predictions).cpu()

    batch_size = keypoints.size(0)
    pck = torch.zeros(batch_size)
    num_pts = torch.zeros(batch_size)
    num_joints = torch.zeros((batch_size, keypoints.size(1)))
    correct_index = -torch.ones((batch_size, len(keypoints[0])))
    for idx in range(batch_size):
        # computes pck for all keypoint pairs of once instance
        p_src = keypoints[idx, :]
        p_pred = predictions[idx, :]
        if pck_type == 'object':
            l_pck = torch.Tensor([torch.max(p_src.max(1)[0] - p_src.min(1)[0])])
        elif pck_type == 'image':
            l_pck = torch.Tensor([image_size])
        # True values in mask indicate the keypoint was present in the dataset
        # Negative values indicate the value was not in the dataset
        mask = torch.ne(p_src[:, 0], 0) * torch.ne(p_src[:, 1], 0)
        num_joints[idx] = mask
        # Sum all available keypoints in the dataset
        N_pts = torch.sum(mask)
        # Set points not present in the dataset to false in source and target points
        p_src[~mask, :] = 0
        p_pred[~mask, :] = 0
        num_pts[idx] = N_pts
        point_distance = torch.pow(torch.sum(torch.pow(p_src - p_pred, 2), 1), 0.5)  # 0.5 means squared!!

        L_pck_mat = l_pck.expand_as(point_distance)  # val -> val, val
        correct_points = torch.le(point_distance, L_pck_mat * thresh)
        correct_points[~mask] = False
        # C_pts = torch.sum(correct_points)
        correct_index[idx, :] = correct_points.view(-1)
        # PCK for the image is divided by the number of valid points in GT
        pck[idx] = torch.sum(correct_points.float()) / torch.clamp(N_pts, min=1e-6)

    # Reduce to joint granularity
    correct_per_joint = torch.sum(correct_index, dim=0)
    sum_available_joint = torch.sum(num_joints, dim=0)
    # clamp the tensor, sometimes we have zero available joints and then we have NaN values
    pck_joints = correct_per_joint / torch.clamp(sum_available_joint, min=1e-6)
    #TODO: joint is not present in the dataset here it will still be returned as "zero" accuracy
    return pck.mean().numpy(), pck_joints.numpy()

# def scale_img(x):
#     """
#     Scale in between 0 and 1
#     :param x:
#     :return:
#     """
#     # ma = torch.max(x)
#     # mi = torch.min(x)
#     out = (x + 1.0) / 2.0
#     out = torch.clamp(out, 0.0, 1.0)
#     return out
#
# class L1LossInstances(torch.nn.L1Loss):
#     """L1Loss, which reduces to instances of the batch
#     """
#
#     def __init__(self):
#         super().__init__(reduction="none")
#
#     def forward(self, image: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
#         super_result = super().forward(image, target)
#         reduced = super_result.mean(axis=(1, 2, 3))
#         return reduced
#

# def latent_kl(prior_mean, posterior_mean):
#     """
#     :param prior_mean:
#     :param posterior_mean:
#     :return:
#     """
#     kl = 0.5 * torch.pow(prior_mean - posterior_mean, 2)
#     kl = torch.sum(kl, dim=[1, 2, 3])
#     # kl = torch.mean(kl)
#
#     return kl
#
#
# def aggregate_kl_loss(prior_means, posterior_means):
#     kl_loss = torch.sum(
#         torch.cat(
#             [
#                 latent_kl(p, q).unsqueeze(dim=-1)
#                 for p, q in zip(
#                     list(prior_means.values()), list(posterior_means.values())
#                 )
#             ],
#             dim=-1,
#         ),
#         dim=-1,
#     )
#     return kl_loss
# def heatmaps_to_coords(heatmaps, thresh: float = 0.0):
#     """
#     Args:
#         heatmaps: numpy.ndarray([batch_size, num_joints, height, width])
#         thresh:
#
#     Returns:
#
#     From: https://github.com/microsoft/human-pose-estimation.pytorch/blob/master/lib/core/inference.py
#     get predictions from score maps
#
#     """
#
#     assert isinstance(heatmaps, np.ndarray), \
#         'batch_heatmaps should be numpy.ndarray'
#     assert heatmaps.ndim == 4, 'batch_images should be 4-ndim'
#
#     batch_size = heatmaps.shape[0]
#     num_joints = heatmaps.shape[1]
#     width = heatmaps.shape[3]
#     heatmaps_reshaped = heatmaps.reshape((batch_size, num_joints, -1))
#     idx = np.argmax(heatmaps_reshaped, 2)
#     maxvals = np.amax(heatmaps_reshaped, 2)
#
#     maxvals = maxvals.reshape((batch_size, num_joints, 1))
#     idx = idx.reshape((batch_size, num_joints, 1))
#
#     preds = np.tile(idx, (1, 1, 2)).astype(np.float32)
#
#     preds[:, :, 0] = (preds[:, :, 0]) % width
#     preds[:, :, 1] = np.floor((preds[:, :, 1]) / width)
#
#     pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 1, 2))
#     pred_mask = pred_mask.astype(np.float32)
#
#     preds *= pred_mask
#     return preds, maxvals


# def hm_to_coords(heatmaps, thresh: float = 0.0):
#     assert isinstance(heatmaps, torch.Tensor), \
#         f'heatmaps should be torch.Tensor, got {type(heatmaps)}'
#     assert len(heatmaps.size()) == 4, f'batch_images should be 4-ndim, got {heatmaps.size}'
#
#     batch_size = heatmaps.size(0)
#     num_joints = heatmaps.size(1)
#     width = heatmaps.size(3)
#     # Flatten input array
#     heatmaps_reshaped = heatmaps.view((batch_size, num_joints, -1))
#     maxvals, idx = torch.max(heatmaps_reshaped, 2)
#
#     maxvals = maxvals.view((batch_size, num_joints, 1))
#     idx = idx.view((batch_size, num_joints, 1))
#
#     # preds = torch.as_tensor(np.tile(idx, (1, 1, 2)) , requires_grad=True).double()
#     preds = idx.repeat((1, 1, 2)).double()
#     preds[:, :, 0] = (preds[:, :, 0]) % width  # TODO MODULO
#     preds[:, :, 1] = torch.floor((preds[:, :, 1]) / width)
#
#     # pred_mask = torch.from_numpy(np.tile(torch.gt(maxvals, 0.0), (1, 1, 2))).double().requires_grad_()
#     pred_mask = torch.gt(maxvals, 0.0).repeat(1, 1, 2).double().requires_grad_()
#     preds *= pred_mask
#     return preds, maxvals
#
#
# def keypoint_loss(predictions, gt_coords, use_torch=False):
#     crit = torch.nn.MSELoss()
#     if use_torch:
#         coords, _ = hm_to_coords(predictions)
#         return crit(coords, torch.from_numpy(gt_coords))
#     else:
#         coords, _ = heatmaps_to_coords(predictions.cpu().detach().numpy())
#         return crit(torch.from_numpy(coords), torch.from_numpy(gt_coords))


# class KeypointLoss(torch.nn.MSELoss):
#     """
#     KeypointLoss based on MSELoss
#     """
#
#     def __init__(self):
#         super().__init__(reduction="none")
#
#     def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
#         predictions = input
#         gt_coors = torch.from_numpy(target)
#         return super().forward(input, target)

# def percentage_correct_keypoints(keypoints: np.array, predicted: np.array, distance: float = 0.5):
#     """
#     Args:
#         keypoints: keypoints with shape [B, N, 2] or [N,2]
#         predicted: predicted keypoints with shape [B, N, 2] or [N,2]
#         distance:
#
#     Returns:
#     """
#     assert type(keypoints) == np.array, f"Keypoints should be type np.array, got {type(keypoints)}"
#     assert type(predicted) == np.array, f"Predicted should be type np.array, got {type(predicted)}"
#     assert type(distance) == float, f"Distance should be type float, got {type(distance)}"
#     assert keypoints.shape == predicted.shape, f"Arrays should have the same shape, got {keypoints.shape} and {predicted.shape}"
#
#     batch_size = keypoints.size(0)
#
#     # Todo for batch
#     keypoints_close = np.isclose(keypoints, predicted, atol=distance)
#     return np.sum(keypoints_close) / len(keypoints)