import torch
import numpy as np
from AnimalPose.utils.image_utils import heatmaps_to_coords
from AnimalPose.data.util import make_heatmaps
import cv2

kpts = np.array(
    [[212.27, 217.87],
     [187.5, 216.06],
     [200.62, 232.],
     [223.27, 194.05],
     [180.77, 190.11],
     [146.64, 246.13],
     [222.77, 233.87],
     [0., 0.],
     [149.97, 146.13],
     [0., 0.],
     [0., 0.],
     [0., 0.],
     [0., 0.],
     [0., 0.],
     [0., 0.],
     [0., 0.],
     [0., 0.],
     [161.875, 159.375],
     [0., 0.],
     [0., 0.]])

hm = make_heatmaps(np.zeros((250, 250)), kpts, 0.1)
hm = np.expand_dims(hm, 0)


def write_tensor_image(tensor):
    for idx, img in enumerate(tensor):
        img = img * 255
        cv2.imwrite(f"hm_orig_{idx}.png", img)


zeros_only = np.array([[0, 0],
                       [0, 0],
                       [0, 0],
                       [0, 0],
                       [0, 0]])

hm_zeros = make_heatmaps(np.zeros((250, 250)), zeros_only, 0.1)
hm_zeros = np.expand_dims(hm_zeros, 0)


# write_tensor_image(hm.squeeze())

def test_heatmaps_to_coords():
    coords, _ = heatmaps_to_coords(hm)
    assert torch.isclose(coords.squeeze().type(torch.int32), torch.from_numpy(kpts).type(torch.int32), atol=1).all()
    # test zeros only
    coords_zeros, _ = heatmaps_to_coords(hm_zeros)
    assert coords_zeros.type(torch.uint8).all() == 0

    coords, _ = heatmaps_to_coords(hm, 0.5)
    assert torch.isclose(coords.squeeze().type(torch.int32), torch.from_numpy(kpts).type(torch.int32), atol=1).all()
    # test zeros only
    coords_zeros, _ = heatmaps_to_coords(hm_zeros, 0.5)
    assert coords_zeros.type(torch.uint8).all() == 0


def test_heatmaps_to_coords_thresholded():
    predictions = np.load("AnimalPose/tests/predictions_000414.npy")
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    for thresh in thresholds:
        coords, _ = heatmaps_to_coords(np.expand_dims(predictions, 0), thresh)

    # write_tensor_image(predictions)
