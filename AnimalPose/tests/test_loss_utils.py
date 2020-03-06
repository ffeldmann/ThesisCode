import numpy as np
from AnimalPose.utils.loss_utils import percentage_correct_keypoints
def test_percentage_correct_keypoints():

    kpts = np.array(([]))
    preds = np.array([])
    assert percentage_correct_keypoints(kpts, preds) == 1
