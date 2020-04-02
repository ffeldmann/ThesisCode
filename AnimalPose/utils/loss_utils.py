import numpy as np
import torch
import torch.nn


def heatmap_loss(targets, predictions):
    crit = torch.nn.MSELoss()#reduction="sum")
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

def heatmaps_to_coords(heatmaps, thresh: float = 0.0):
    """
    Args:
        heatmaps: numpy.ndarray([batch_size, num_joints, height, width])
        thresh:

    Returns:

    From: https://github.com/microsoft/human-pose-estimation.pytorch/blob/master/lib/core/inference.py
    get predictions from score maps

    """

    assert isinstance(heatmaps, np.ndarray), \
        'batch_heatmaps should be numpy.ndarray'
    assert heatmaps.ndim == 4, 'batch_images should be 4-ndim'

    batch_size = heatmaps.shape[0]
    num_joints = heatmaps.shape[1]
    width = heatmaps.shape[3]
    heatmaps_reshaped = heatmaps.reshape((batch_size, num_joints, -1))
    idx = np.argmax(heatmaps_reshaped, 2)
    maxvals = np.amax(heatmaps_reshaped, 2)

    maxvals = maxvals.reshape((batch_size, num_joints, 1))
    idx = idx.reshape((batch_size, num_joints, 1))

    preds = np.tile(idx, (1, 1, 2)).astype(np.float32)

    preds[:, :, 0] = (preds[:, :, 0]) % width
    preds[:, :, 1] = np.floor((preds[:, :, 1]) / width)

    pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 1, 2))
    pred_mask = pred_mask.astype(np.float32)

    preds *= pred_mask
    return preds, maxvals


def hm_to_coords(heatmaps, thresh: float = 0.0):
    assert isinstance(heatmaps, torch.Tensor), \
        f'heatmaps should be torch.Tensor, got {type(heatmaps)}'
    assert len(heatmaps.size()) == 4, f'batch_images should be 4-ndim, got {heatmaps.size}'

    batch_size = heatmaps.size(0)
    num_joints = heatmaps.size(1)
    width = heatmaps.size(3)
    # Flatten input array
    heatmaps_reshaped = heatmaps.view((batch_size, num_joints, -1))
    maxvals, idx = torch.max(heatmaps_reshaped, 2)

    maxvals = maxvals.view((batch_size, num_joints, 1))
    idx = idx.view((batch_size, num_joints, 1))

    # preds = torch.as_tensor(np.tile(idx, (1, 1, 2)) , requires_grad=True).double()
    preds = idx.repeat((1, 1, 2)).double()
    preds[:, :, 0] = (preds[:, :, 0]) % width  # TODO MODULO
    preds[:, :, 1] = torch.floor((preds[:, :, 1]) / width)

    # pred_mask = torch.from_numpy(np.tile(torch.gt(maxvals, 0.0), (1, 1, 2))).double().requires_grad_()
    pred_mask = torch.gt(maxvals, 0.0).repeat(1, 1, 2).double().requires_grad_()
    preds *= pred_mask
    return preds, maxvals


def keypoint_loss(predictions, gt_coords, use_torch=False):
    crit = torch.nn.MSELoss()
    if use_torch:
        coords, _ = hm_to_coords(predictions)
        return crit(coords, torch.from_numpy(gt_coords))
    else:
        coords, _ = heatmaps_to_coords(predictions.cpu().detach().numpy())
        return crit(torch.from_numpy(coords), torch.from_numpy(gt_coords))


class KeypointLoss(torch.nn.MSELoss):
    """
    KeypointLoss based on MSELoss
    """

    def __init__(self):
        super().__init__(reduction="none")

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        predictions = input
        gt_coors = torch.from_numpy(target)
        return super().forward(input, target)


class MSELossInstances(torch.nn.MSELoss):
    """MSELoss, which reduces to instances of the batch
    """

    def __init__(self):
        super().__init__(reduction="none")

    def forward(self, image: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        super_result = super().forward(image, target)
        reduced = super_result.mean(axis=(1, 2, 3))
        return reduced


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


def percentage_correct_keypoints(keypoints, predictions, alpha=0.1):
    """

    Args:
        keypoints:
        predictions:
        alpha:

    Returns: pck mean, pck per joint

    """
    keypoints = torch.from_numpy(keypoints)
    predictions = torch.from_numpy(predictions)
    batch_size = keypoints.size(0)
    pck = torch.zeros((batch_size))
    num_pts = torch.zeros((batch_size))
    num_joints = torch.zeros((batch_size, keypoints.size(1)))
    correct_index = -torch.ones((batch_size, len(keypoints[0])))
    for idx in range(batch_size):
        # computes pck for all keypoint pairs of once instance
        p_src = keypoints[idx, :]
        p_pred = predictions[idx, :]
        # if dataset_name == 'PF-WILLOW':
        #L_pck = torch.Tensor([torch.max(p_src.max(1)[0] - p_src.min(1)[0])])
        # elif dataset_name == 'PF-PASCAL':
        L_pck = torch.Tensor([128.0])
        #N_pts = torch.sum(torch.ne(p_src[:, 0], 0) * torch.ne(p_src[:, 1], 0))
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
        #point_distance = torch.pow(torch.sum(torch.pow(p_src[, :N_pts] - p_pred[:, :N_pts], 2), 0), 0.5)
        point_distance = torch.pow(torch.sum(torch.pow(p_src - p_pred, 2), 1), 0.5) # 0.5 means squared!!

        L_pck_mat = L_pck.expand_as(point_distance)  # val -> val, val
        correct_points = torch.le(point_distance, L_pck_mat * alpha)
        correct_points[~mask] = False
        #C_pts = torch.sum(correct_points)
        correct_index[idx, :] = correct_points.view(-1)
        # PCK for the image is divided by the number of valid points in GT
        pck[idx] = torch.sum(correct_points.float()) / N_pts
    # TODO
    # batch_size is 1
    #if batch_size == 1:
    #    pck = pck[0].item()
    #    num_pts = int(num_pts[0].item())
    #    correct_index = correct_index.squeeze().cpu().numpy().astype(np.int8)
    #    correct_index = correct_index[np.where(correct_index > -1)]

    # Reduce to joint granularity
    correct_per_joint = torch.sum(correct_index, dim=0)
    sum_available_joint = torch.sum(num_joints, dim=0)
    pck_joints = correct_per_joint / sum_available_joint
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

# class MaskedL1LossInstances(L1LossInstances):
#     def __init__(self, config: dict) -> None:
#         super().__init__()
#         self.config = config
#         self.mask_creator = 0
#
#     def forward(
#         self,
#         image: torch.Tensor,
#         target: torch.Tensor,
#         forward_flow: torch.Tensor,
#         backward_flow: torch.Tensor,
#     ) -> torch.Tensor:
#         _, masked_image, masked_target = self.get_masked(
#             image, target, forward_flow, backward_flow
#         )
#
#         return 0
#
#     def get_masked(
#         self,
#         image: torch.Tensor,
#         target: torch.Tensor,
#         forward_flow: torch.Tensor,
#         backward_flow: torch.Tensor,
#     ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
#         mask = self.mask_creator(forward_flow, backward_flow)
#         masked_image = mask * image
#         masked_target = mask * target
#         return mask, masked_image, masked_target


# class MaskedL1LossInstancesForwardFlow(L1LossInstances):
#     def __init__(self, config: dict) -> None:
#         super().__init__()
#         self.config = config
#         self.mask_creator_forward = MaskCreatorForward(
#             self.config["mask_sigma"], self.config["mask_threshold"]
#         )
#
#     def forward(
#         self, image: torch.Tensor, target: torch.Tensor, forward_flow: torch.Tensor,
#     ) -> torch.Tensor:
#         _, masked_image, masked_target = self.get_masked(image, target, forward_flow)
#         loss = super().forward(masked_image, masked_target)
#         return loss
#
#     def get_masked(
#         self, image: torch.Tensor, target: torch.Tensor, forward_flow: torch.Tensor,
#     ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
#         mask = self.mask_creator_forward(forward_flow)
#         masked_image = mask * image
#         masked_target = mask * target
#         return mask, masked_image, masked_target


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
