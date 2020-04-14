import numpy as np
import torch

from AnimalPose.utils.loss_utils import percentage_correct_keypoints

kpts = np.array([[265.28, 248.07],
                     [171.58, 240.8],
                     [213.37, 300.79],
                     [307.91, 165.94],
                     [138.55, 153.93],
                     [0., 0.],
                     [0., 0.],
                     [64.47, 416.01],
                     [160.94, 346.32],
                     [0., 0.],
                     [205.49, 112.59],
                     [0., 0.],
                     [0., 0.],
                     [0., 0.],
                     [0., 0.],
                     [0., 0.],
                     [0., 0.],
                     [14.0625, 361.25],
                     [0., 0.],
                     [0., 0.]])
kpts = np.expand_dims(kpts, 0)
preds = kpts

def test_pck_same_input():

    pckmean, pckperjoint = percentage_correct_keypoints(kpts, preds)
    assert pckmean == 1
    assert pckperjoint.all() == np.array([1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0]).all()

def test_pck_thresholds():
    # Test different thresholds
    arr = np.array([[100, 50],
                    [0., 0.],
                    [1, 1],
                    [99, 70],
                    [55, 55],
                    [0, 0],
                    [0, 0],
                    # [1, 1],
                    ])

    arr = np.expand_dims(arr, 0)

    arr2 = np.array([[23, 23],
                     [0., 0.],
                     [1, 1],
                     [98, 70],
                     [55, 54],
                     [5, 5],
                     [100, 100],
                     # [0, 0],
                     ])
    arr2 = np.expand_dims(arr2, 0)

    pckmean, pckperjoint = percentage_correct_keypoints(arr, arr2, thresh=0.5)

    assert pckperjoint.all() == np.array([0, 0, 1, 1, 1, 0, 0]).all()
    assert pckmean == 0.75, f"Got {pckmean}"

    pckmean, pckperjoint = percentage_correct_keypoints(arr, arr2, thresh=1)
    assert pckmean == 1
    assert pckperjoint.all() == np.array([1, 0, 1, 1, 1, 0, 0]).all()
