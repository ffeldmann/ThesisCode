import numpy as np
import torch


def heatmap_loss(targets, predictions):
    crit = torch.nn.MSELoss(reduction="sum")
    return crit(torch.from_numpy(targets), predictions)


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


def keypoint_loss(predictions, gt_coords):
    crit = torch.nn.MSELoss()
    coords, _ = heatmaps_to_coords(predictions.cpu().detach().numpy())
    return crit(torch.from_numpy(coords), torch.from_numpy(gt_coords))


class MSELossInstances(torch.nn.MSELoss):
    """MSELoss, which reduces to instances of the batch
    """

    def __init__(self):
        super().__init__(reduction="none")

    def forward(self, image: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        super_result = super().forward(image, target)
        reduced = super_result.mean(axis=(1, 2, 3))
        return reduced


def percentage_correct_keypoints(keypoints: np.array, predicted: np.array, distance: float = 0.5):
    """
    Args:
        keypoints: keypoints with shape [B, N, 2] or [N,2]
        predicted: predicted keypoints with shape [B, N, 2] or [N,2]
        distance:

    Returns:
    """
    assert type(keypoints) == np.array, f"Keypoints should be type np.array, got {type(keypoints)}"
    assert type(predicted) == np.array, f"Predicted should be type np.array, got {type(predicted)}"
    assert type(distance) == float, f"Distance should be type float, got {type(distance)}"

    assert keypoints.shape == predicted.shape, f"Arrays should have the same shape, got {keypoints.shape} and {predicted.shape}"

    # Todo for batch
    keypoints_close = np.isclose(keypoints, predicted, atol=distance)
    return np.sum(keypoints_close) / len(keypoints)




def pck(source_points, warped_points, dataset_name='PF-PASCAL', alpha=0.1):
    batch_size = source_points.size(0)
    pck = torch.zeros((batch_size))
    num_pts = torch.zeros((batch_size))
    correct_index = -torch.ones((batch_size, 20))
    for idx in range(batch_size):
        p_src = source_points[idx, :]
        p_wrp = warped_points[idx, :]
        if dataset_name == 'PF-WILLOW':
            L_pck = torch.Tensor([torch.max(p_src.max(1)[0] - p_src.min(1)[0])]).cuda()
        elif dataset_name == 'PF-PASCAL':
            L_pck = torch.Tensor([224.0]).cuda()
        N_pts = torch.sum(torch.ne(p_src[0, :], -1) * torch.ne(p_src[1, :], -1))
        num_pts[idx] = N_pts
        point_distance = torch.pow(torch.sum(torch.pow(p_src[:, :N_pts] - p_wrp[:, :N_pts], 2), 0), 0.5)
        L_pck_mat = L_pck.expand_as(point_distance)
        correct_points = torch.le(point_distance, L_pck_mat * alpha)
        C_pts = torch.sum(correct_points)
        correct_index[idx, :C_pts] = torch.nonzero(correct_points).view(-1)
        pck[idx] = torch.mean(correct_points.float())

    # batch_size is 1
    if batch_size == 1:
        pck = pck[0].item()
        num_pts = int(num_pts[0].item())
        correct_index = correct_index.squeeze().cpu().numpy().astype(np.int8)
        correct_index = correct_index[np.where(correct_index > -1)]

    return pck, correct_index, num_pts







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
