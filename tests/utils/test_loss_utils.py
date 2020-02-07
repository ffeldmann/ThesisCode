import torch
import numpy as np
from AnimalPose.utils.loss_utils import heatmaps_to_coords, heatmap_loss, keypoint_loss
from AnimalPose.scripts.load_config import load_config

kpts = np.array(
        [[212.27 , 217.87 ],
        [187.5  , 216.06 ],
        [200.62 , 232.   ],
        [223.27 , 194.05 ],
        [180.77 , 190.11 ],
        [146.64 , 246.13 ],
        [222.77 , 263.87 ],
        [  0.   ,   0.   ],
        [199.97 , 246.13 ],
        [  0.   ,   0.   ],
        [  0.   ,   0.   ],
        [  0.   ,   0.   ],
        [  0.   ,   0.   ],
        [  0.   ,   0.   ],
        [  0.   ,   0.   ],
        [  0.   ,   0.   ],
        [  0.   ,   0.   ],
        [161.875, 259.375],
        [  0.   ,   0.   ],
        [  0.   ,   0.   ]])


def test_heatmaps_to_coords():
    pass

def test_heatmap_loss():
    pass

def test_keypoint_loss():
    pass