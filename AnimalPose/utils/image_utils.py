import torch
import numpy as np
from AnimalPose.utils.tensor_utils import sure_to_torch, sure_to_numpy
from edflow import get_logger
from edflow.data.util import adjust_support


def heatmaps_to_image(batch_heatmaps: np.ndarray):
    """
    Args:
        batch_heatmaps: Batch of heatmaps of shape [B, C, H, W], C == NUM_JOINTS

    Returns: Batch of images containing heatmaps of shape [B, 1, H, W]

    """
    # batch_heatmaps = sure_to_numpy(batch_heatmaps)
    # https://github.com/numpy/numpy/issues/9568
    np.seterr(under='ignore', invalid='ignore')
    batch, _, width, height = batch_heatmaps.shape
    images = np.sum(batch_heatmaps, axis=1).reshape(batch, 1, height, width)
    # assert images.max() <= batch + 0.5, "Maximum value cannot be possible, something is wrong."

    hm_min = images.min(axis=(1, 2, 3), keepdims=True)
    hm_max = images.max(axis=(1, 2, 3), keepdims=True)
    hm_max.clip(min=1e-6)
    images = (images - hm_min) / (hm_max - hm_min)
    return images


def gauss(x, a, b, c, d=0):
    # Helper function for color_heatmap
    return a * np.exp(-(x - b) ** 2 / (2 * c ** 2)) + d


def color_heatmap(heatmap: np.array):
    """
    Creats a colored heatmap of a single heatmap array.
    Args:
        heatmap: np.array of shape [H,W,C]
    Returns: np.array of shape [H,W,C] as np.unit8

    """
    # assert heatmap.shape[2] != 1, f"Need [H,W,C] , got {heatmap.shape}"
    heatmap = sure_to_torch(heatmap).squeeze()
    color = np.zeros((heatmap.shape[0], heatmap.shape[1], 3))
    color[:, :, 0] = gauss(heatmap, .5, .6, .2) + gauss(heatmap, 1, .8, .3)
    color[:, :, 1] = gauss(heatmap, 1, .5, .3)
    color[:, :, 2] = gauss(heatmap, 1, .2, .3)
    color[color > 1] = 1
    color = (color * 255).astype(np.uint8)
    return color


def get_color_heatmaps(heatmaps: np.array):
    """

    Args:
        heatmaps: np.array of shape [B, C, H, W]

    Returns: np.array of shape [B, 1, H, W]

    """
    batches, joints, width, height = heatmaps.shape
    heat_out = np.zeros((batches, 3, width, height))
    images = heatmaps_to_image(heatmaps).transpose(0, 2, 3, 1)
    for idx in range(batches):
        heat_out[idx, :, :, :] = color_heatmap(images[idx]).transpose(2, 0, 1)
    return heat_out


def normalize_nparray(array: np.array):
    """
        Normalize array to 0->1
        Args:
            array:

        Returns:

        """
    array -= array.min(1, keepdims=True)[0]
    array /= array.max(1, keepdims=True)[0].clip(min=1e-6)
    return array


def normalize_tensor(tensor: torch.tensor):
    """
    Normalize tensor to 0->1
    Args:
        tensor:

    Returns:

    """

    tensor -= tensor.min(1, keepdim=True)[0]
    tensor /= tensor.max(1, keepdim=True)[0]
    return tensor


def apply_threshold_to_heatmaps(heatmaps: torch.tensor, thresh: float):
    """

    Args:
        heatmaps: Tensor of shape [N, 1, H, W]
        thresh: Threshold for a keypoint in a heatmap to be considered a heatmap
                thresh should be in range [0,1]
        inplace: Performs the operation inplace

    Returns: Thresholded heatmap

    """
    heatmaps = normalize_tensor(heatmaps)
    if thresh != None:
        assert thresh > 0 or thresh < 1, f"Thresh must be in range [0, 1], got {thresh}"
        # Get the indices where the values are smaller then the threshold
        indices = heatmaps < thresh
        # Set these values to 0
        heatmaps[indices] = 0
    return heatmaps


def heatmaps_to_coords(heatmaps: torch.tensor, thresh: float = None):
    """
    Get predictions from heatmaps in torch Tensor.

    Args:
        heatmaps: Tensor of shape [N, 1, H, W]
        thresh: Threshold for a keypoint in a heatmap to be considered a heatmap
                thresh should be in range [0,1]

    Returns: torch.LongTensor

    """
    heatmaps = sure_to_torch(heatmaps)
    #heatmaps = normalize_tensor(heatmaps) # evil if you dont clone the tensor before!
    if thresh != None:
        #TODO assert correct!
        assert thresh > 0 or thresh < 1, f"Thresh must be in range [0, 1], got {thresh}"
        # Get the indices where the values are smaller then the threshold
        indices = heatmaps < thresh
        # Set these values to 0
        heatmaps[indices] = 0
    assert heatmaps.dim() == 4, 'Heatmaps should be 4-dim'
    maxval, idx = torch.max(heatmaps.view(heatmaps.size(0), heatmaps.size(1), -1), 2)

    maxval = maxval.view(heatmaps.size(0), heatmaps.size(1), 1)
    idx = idx.view(heatmaps.size(0), heatmaps.size(1), 1) + 1

    preds = idx.repeat(1, 1, 2).float()

    preds[:, :, 0] = (preds[:, :, 0] - 1) % heatmaps.size(3) + 1
    preds[:, :, 1] = torch.floor((preds[:, :, 1] - 1) / heatmaps.size(3)) + 1

    pred_mask = maxval.gt(0).repeat(1, 1, 2).float()
    preds *= pred_mask

    return preds
