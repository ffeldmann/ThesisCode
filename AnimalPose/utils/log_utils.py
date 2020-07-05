import matplotlib.pyplot as plt
import numpy as np
from edflow.data.util import adjust_support
from AnimalPose.utils.tensor_utils import sure_to_torch, sure_to_numpy
import skimage.feature
import skimage.exposure
import scipy
import cv2
import torch


def generate_samples(input, network, n_samples=5):
    images_torch = input[:n_samples]
    sigmoid = torch.nn.Sigmoid()
    blank = torch.ones_like(images_torch[0])
    output = [torch.cat([blank] + list(images_torch), dim=2)]
    for i in range(images_torch.size(0)):
        converted_imgs = [images_torch[i]]
        predictions, _, _ = network(images_torch, torch.cat(n_samples * [images_torch[i][None, ...]]))
        predictions = sigmoid(predictions)
        for j in range(predictions.size(0)):
            converted_imgs.append(predictions[j])
        output.append(torch.cat(converted_imgs, dim=2))
    return np.array(torch.cat(output, dim=1).detach().cpu()).transpose(1, 2, 0)


def hist_similarity(img1, img2, PLOT=False):
    img1 = adjust_support(img1, "0->255", "0->1")
    img2 = adjust_support(img2, "0->255", "0->1")

    OPENCV_METHODS = (
        ("Correlation", cv2.HISTCMP_CORREL),
        ("Chi-Squared", cv2.HISTCMP_CHISQR),
        ("Intersection", cv2.HISTCMP_INTERSECT),
        ("Hellinger", cv2.HISTCMP_BHATTACHARYYA))

    hist1 = cv2.calcHist([img1], [0, 1, 2], None, [8, 8, 8],
                         [0, 256, 0, 256, 0, 256])
    hist1 = cv2.normalize(hist1, hist1).flatten()

    hist2 = cv2.calcHist([img2], [0, 1, 2], None, [8, 8, 8],
                         [0, 256, 0, 256, 0, 256])
    hist2 = cv2.normalize(hist2, hist2).flatten()

    hist_compare = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)

    if PLOT:
        fig, (ax1, ax2) = plt.subplots(1, 2)
        plt.title(f"histcompare: {hist_compare}")
        ax1.imshow(img1)
        ax2.imshow(img2)
        plt.show()

    return hist_compare


def hog_similarity(img1, img2=False, PLOT=False):
    orientations = 8
    pixels_per_cell = (16, 16)
    img1 = adjust_support(img1, "0->255", "0->1")
    hog_inp, hog_img_inp = skimage.feature.hog(img1,
                                               orientations=orientations,
                                               pixels_per_cell=pixels_per_cell,
                                               visualize=True,
                                               feature_vector=True)
    hog_img_inp = skimage.exposure.rescale_intensity(hog_img_inp, in_range=(0, 10))

    if type(img2) == np.ndarray:
        img2 = adjust_support(img2, "0->255", "0->1")
        hog_dis, hog_img_dis = skimage.feature.hog(img2,
                                                   orientations=orientations,
                                                   pixels_per_cell=pixels_per_cell,
                                                   # cells_per_block=(1, 1),
                                                   visualize=True,
                                                   feature_vector=True)
        hog_img_dis = skimage.exposure.rescale_intensity(hog_img_dis, in_range=(0, 10))

        value = 1 - scipy.spatial.distance.cosine(hog_inp, hog_dis)
        if PLOT:
            fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4)
            plt.title(value)
            ax1.imshow(np.stack((hog_img_inp,) * 3, axis=0).transpose(1, 2, 0)[:, :, 0])
            ax2.imshow(hog_img_dis)
            ax3.imshow(adjust_support(img1, "0->1"))
            ax4.imshow(adjust_support(img2, "0->1"))
            plt.show()
        return value, [hog_img_inp, hog_img_dis]

    return hog_img_inp


def plot_pred_figure(images, predictions, labels=None):
    """
    Remember to clip output numpy array to [0, 255] range and cast it to uint8.
    Otherwise matplot.pyplot.imshow would show weird results.
    Args:
        images:
        predictions:
    Returns:
    """
    from AnimalPose.data.animals_VOC2011 import animal_class
    idx_to_animal = {v: k for k, v in animal_class.items()}
    fig = plt.figure(figsize=(10, 10))
    for idx in range(8):
        fig.add_subplot(4, 2, idx + 1)
        fig.suptitle('Input, Prediction')
        if labels != None:
            plt.title(f"GT:{labels[idx]}, Pred: {predictions[idx]}")
        else:
            plt.title(f"{idx_to_animal[predictions[idx]]}")
        plt.imshow(adjust_support(images[idx].cpu().numpy().transpose(1, 2, 0), "0->1", "0->1"))
        plt.tight_layout()
    return fig


def plot_input_target_keypoints(inputs: np.ndarray, targets, gt_coords, coords):
    """
    Remember to clip output numpy array to [0, 255] range and cast it to uint8.
     Otherwise matplot.pyplot.imshow would show weird results.
    Args:
        inputs:
        targets:
        gt_coords:

    Returns:

    """
    fig = plt.figure(figsize=(10, 10))
    # heatmaps_to_coords needs [batch_size, num_joints, height, width]
    # coords, _ = heatmaps_to_coords(targets)
    coords = sure_to_numpy(coords.clone())
    for idx in range(8):
        fig.add_subplot(4, 2, idx + 1)
        fig.suptitle('Blue: GT, Red: Predicted')
        if inputs[idx].shape[-1] == 1:
            plt.imshow(adjust_support(inputs[idx].squeeze(-1), "0->255"))
        else:
            plt.imshow(adjust_support(inputs[idx], "0->255"))
        mask = np.ones(20).astype(bool)
        for kpt in range(0, len(coords[0])):
            if (gt_coords[idx][:, :2][kpt] == [0, 0]).all():
                mask[kpt] = False
                # If gt_coords are 0,0 meaning not present in the dataset, don't draw them.
                continue

            plt.plot([np.array(gt_coords[idx][:, :2][kpt][0]),
                      np.array(coords[idx][kpt][0])],
                     [np.array(gt_coords[idx][:, :2][kpt][1]),
                      np.array(coords[idx][kpt][1])],
                     'bx-', alpha=0.3)

        plt.scatter(gt_coords[idx][mask][:, 0],
                    gt_coords[idx][mask][:, 1],
                    c="blue")
        plt.scatter(coords[idx][mask][:, 0],
                    coords[idx][mask][:, 1],
                    c="red")
    return fig
