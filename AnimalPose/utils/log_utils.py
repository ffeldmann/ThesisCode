import matplotlib.pyplot as plt
import numpy as np
from AnimalPose.utils.loss_utils import heatmaps_to_coords

def plot_input_target_keypoints(inputs, targets, gt_coords):
    fig = plt.figure(figsize=(10, 10))
    # heatmaps_to_coords needs [batch_size, num_joints, height, width]
    coords, _ = heatmaps_to_coords(targets)
    for idx in range(4):
        fig.add_subplot(2, 2, idx + 1)
        fig.suptitle('Blue: GT, Red: Predicted')
        if inputs[idx].shape[-1] == 1:
            plt.imshow(inputs[idx].squeeze(-1))
        else:
            plt.imshow(inputs[idx])
        for kpt in range(0, len(coords[0])):
            if (gt_coords[idx][:, :2][kpt] == [0,0]).all():
                # If gt_coords are 0,0 meaning not present in the dataset, don't draw them.
                continue
            plt.plot([np.array(gt_coords[idx][:, :2][kpt][0]),
                      np.array(coords[idx][kpt][0])],
                     [np.array(gt_coords[idx][:, :2][kpt][1]),
                      np.array(coords[idx][kpt][1])],
                     'bx-', alpha=0.3)
            plt.scatter(gt_coords[idx][:, 0],
                    gt_coords[idx][:, 1],
                    c="blue")
            plt.scatter(coords[idx][:, 0],
                        coords[idx][:, 1],
                        c="red")
    return fig