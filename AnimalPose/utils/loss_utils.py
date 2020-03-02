import numpy as np
import torch


def heatmap_loss(targets, predictions):
    crit = torch.nn.MSELoss(reduction="sum")
    return crit(torch.from_numpy(targets), predictions)


def heatmaps_to_coords(heatmaps):
    """
    Args:
        heatmaps: numpy.ndarray([batch_size, num_joints, height, width])

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


class L1LossInstances(torch.nn.L1Loss):
    """L1Loss, which reduces to instances of the batch
    """

    def __init__(self):
        super().__init__(reduction="none")

    def forward(self, image: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        super_result = super().forward(image, target)
        reduced = super_result.mean(axis=(1, 2, 3))
        return reduced


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


def scale_img(x):
    """
    Scale in between 0 and 1
    :param x:
    :return:
    """
    # ma = torch.max(x)
    # mi = torch.min(x)
    out = (x + 1.0) / 2.0
    out = torch.clamp(out, 0.0, 1.0)
    return out


# VGGOutput = namedtuple(
#     "VGGOutput", ["input", "relu1_2", "relu2_2", "relu3_2", "relu4_2", "relu5_2"],
# )
# VGGTargetLayers = {
#     "3": "relu1_2",
#     "8": "relu2_2",
#     "13": "relu3_2",
#     "22": "relu4_2",
#     "31": "relu5_2",
# }
#
#
# def vgg_loss(custom_vgg, target, pred, weights=None):
#     """
#
#     :param custom_vgg:
#     :param target:
#     :param pred:
#     :return:
#     """
#     target_feats = custom_vgg(target)
#     pred_feats = custom_vgg(pred)
#     if weights is None:
#
#         loss = torch.cat(
#             [
#                 FLAGS.vgg_feat_weights[i]
#                 * torch.mean(torch.abs(tf - pf), dim=[1, 2, 3]).unsqueeze(dim=-1)
#                 for i, (tf, pf) in enumerate(zip(target_feats, pred_feats))
#             ],
#             dim=-1,
#         )
#     else:
#         pix_loss = [
#             FLAGS.vgg_feat_weights[0]
#             * torch.mean(weights * torch.abs(target_feats[0] - pred_feats[0]))
#             .unsqueeze(dim=-1)
#             .to(torch.float)
#         ]
#         loss = torch.cat(
#             pix_loss
#             + [
#                 FLAGS.vgg_feat_weights[i + 1]
#                 * torch.mean(torch.abs(tf - pf), dim=[1, 2, 3]).unsqueeze(dim=-1)
#                 for i, (tf, pf) in enumerate(zip(target_feats[1:], pred_feats[1:]))
#             ],
#             dim=-1,
#         )
#
#     loss = torch.sum(loss, dim=1)
#     return loss
#
#
# class PerceptualVGG(torch.nn.Module):
#     def __init__(self, vgg):
#         super().__init__()
#         # self.vgg = vgg19(pretrained=True)
#         if isinstance(vgg, torch.nn.DataParallel):
#             self.vgg_layers = vgg.module.features
#         else:
#             self.vgg_layers = vgg.features
#
#         self.input_transform = transforms.Compose(
#             [
#                 transforms.Normalize(
#                     mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
#                 )
#             ]
#         )
#
#     def forward(self, x):
#         out = {"input": x}
#         # normalize in between 0 and 1
#         x = scale_img(x)
#         # normalize appropriate for vgg
#         x = torch.stack([self.input_transform(el) for el in torch.unbind(x)])
#
#         for name, submodule in self.vgg_layers._modules.items():
#             # x = submodule(x)
#             if name in VGGTargetLayers:
#                 x = submodule(x)
#                 out[VGGTargetLayers[name]] = x
#             else:
#                 with torch.no_grad():
#                     x = submodule(x)
#
#         return VGGOutput(**out)
#
#
# class PerceptualLossInstances(torch.nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.config = config
#         vgg = vgg19(pretrained=True)
#         vgg.eval()
#         self.custom_vgg = PerceptualVGG(vgg)
#
#         vgg_feat_weights = config.setdefault(
#             "vgg_feat_weights", (len(VGGTargetLayers) + 1) * [1.0]
#         )
#         assert len(vgg_feat_weights) == len(VGGTargetLayers) + 1
#         flags.DEFINE_list(
#             "vgg_feat_weights",
#             vgg_feat_weights,
#             "The weights for the considered layer outputs of vgg19 for the perceptual loss.",
#         )
#         FLAGS([""])
#
#     def forward(self, image, target):
#         loss = vgg_loss(self.custom_vgg, target, image)
#         return loss
