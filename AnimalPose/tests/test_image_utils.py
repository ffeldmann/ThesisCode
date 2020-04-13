import torch
import numpy as np
from AnimalPose.utils.image_utils import heatmaps_to_coords
from AnimalPose.data.util import make_heatmaps

kpts = np.array(
    [[212.27, 217.87],
     [187.5, 216.06],
     [200.62, 232.],
     [223.27, 194.05],
     [180.77, 190.11],
     [146.64, 246.13],
     [222.77, 263.87],
     [0., 0.],
     [199.97, 246.13],
     [0., 0.],
     [0., 0.],
     [0., 0.],
     [0., 0.],
     [0., 0.],
     [0., 0.],
     [0., 0.],
     [0., 0.],
     [161.875, 259.375],
     [0., 0.],
     [0., 0.]])
hm = make_heatmaps(np.zeros((250, 250)), kpts, 2)
hm = np.expand_dims(hm, 0)


def test_heatmaps_to_coords():
    coords = heatmaps_to_coords(hm)

    # test zeros only
    zeros_only = np.array([0., 0.],
                          [0., 0.],
                          [0., 0.],
                          [0., 0.],
                          [0., 0.])

    hm =
