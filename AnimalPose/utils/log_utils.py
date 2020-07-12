import matplotlib.pyplot as plt
import numpy as np
from edflow.data.util import adjust_support
from AnimalPose.utils.tensor_utils import sure_to_torch, sure_to_numpy
import skimage.feature
import skimage.exposure
import scipy
import cv2
import torch
from datetime import datetime


def noisy(noise_typ, image):
    if noise_typ == "gauss":
        row, col, ch = image.shape
        mean = 1
        var = 0.1
        sigma = var ** 0.5
        gauss = np.random.normal(mean, sigma, (row, col, ch))
        gauss = gauss.reshape(row, col, ch)
        noisy = gauss
        return noisy
    elif noise_typ == "salt":
        row, col, ch = image.shape
        s_vs_p = 0.5
        amount = 0.004
        out = np.copy(image)
        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
                  for i in image.shape]
        out[coords] = 1

        # Pepper mode
        num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
                  for i in image.shape]
        out[coords] = 0
        return out
    elif noise_typ == "poisson":
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(image * vals) / float(vals)
        return noisy

    elif noise_typ == "speckle":
        row, col, ch = image.shape
        gauss = np.random.randn(row, col, ch)
        gauss = gauss.reshape(row, col, ch)
        noisy = image + image * gauss
        return noisy


# def generate_samples(input, network, n_samples=5):
#     images_torch = input[:n_samples]
#     sigmoid = torch.nn.Sigmoid()
#     blank = torch.ones_like(images_torch[0])
#     output = [torch.cat([blank] + list(images_torch), dim=2)]
#     for i in range(images_torch.size(0)):
#         converted_imgs = [images_torch[i]]
#         predictions, _, _ = network(images_torch, torch.cat(n_samples * [images_torch[i][None, ...]]))
#         predictions = sigmoid(predictions)
#         for j in range(predictions.size(0)):
#             converted_imgs.append(predictions[j])
#         output.append(torch.cat(converted_imgs, dim=2))
#     return np.array(torch.cat(output, dim=1).detach().cpu()).transpose(1, 2, 0)


def generate_samples(input, network, n_samples=4, appreance_random=False, pose_random=False):
    images_torch = input[:n_samples]

    blank = torch.ones_like(images_torch[0])
    random_image = noisy("gauss", np.zeros((128, 128, 3))).transpose(2, 0, 1)
    random_image += abs(random_image.min())
    random_image /= random_image.max()
    random_image = np.clip(random_image, 0, 1)
    random_image = torch.from_numpy(random_image).cuda().float()
    np_blank = np.ones((128, 128))
    if pose_random:
        output = [torch.cat([blank] + list(torch.cat(n_samples * [random_image[None, ...]])), dim=2)]
    else:
        output = [torch.cat([blank] + list(images_torch), dim=2)]
    images_hog = [hog_similarity(images_torch[i].cpu().detach().numpy().transpose(1, 2, 0)) for i in
                  range(images_torch.size(0))]
    output_hog = [np.concatenate([np_blank] + images_hog, axis=1)]
    cosine_distances = []
    cosine_row_wise = []
    hist_similarities = []
    hist_column_wise = []
    for i in range(images_torch.size(0)):
        converted_imgs = [images_torch[i]]

        # hog_imgs = [skimage.feature.hog(images_torch[i].cpu().detach().numpy().transpose(1, 2, 0),
        #                                visualize=True,
        #                                feature_vector=True)[1]]
        # hog_imgs = [np_blank]
        hog_imgs = [hog_similarity(images_torch[i].cpu().detach().numpy().transpose(1, 2, 0))]
        if appreance_random:
            converted_imgs = [random_image]
        # pose, appearance
        if appreance_random:
            predictions, _, _ = network(images_torch, torch.cat(n_samples * [random_image[None, ...]]))
        elif pose_random:
            predictions, _, _ = network(torch.cat(n_samples * [random_image[None, ...]]),
                                        torch.cat(n_samples * [images_torch[i].unsqueeze(0), ]))
        else:
            predictions, _, _ = network(images_torch, torch.cat(n_samples * [images_torch[i][None, ...]]))
        predictions = torch.sigmoid(predictions)
        predictions = predictions.clamp(0.0, 1)
        for j in range(predictions.size(0)):
            converted_imgs.append(predictions[j])
            hog_values = hog_similarity(images_torch[j].cpu().numpy().transpose(1, 2, 0),
                                        predictions[j].cpu().detach().numpy().transpose(1, 2, 0))
            hog_imgs.append(hog_values[1][1])
            cosine_distances.append(hog_values[0])
            hog_row = hog_similarity(images_torch[i].cpu().numpy().transpose(1, 2, 0),
                                     predictions[j].cpu().detach().numpy().transpose(1, 2, 0))
            cosine_row_wise.append(hog_row[0])

            hist_similarities.append(hist_similarity(images_torch[i].cpu().numpy().transpose(1, 2, 0),
                                                     predictions[j].cpu().detach().numpy().transpose(1, 2, 0)))

            hist_column_wise.append(hist_similarity(images_torch[j].cpu().numpy().transpose(1, 2, 0),
                                                    predictions[j].cpu().detach().numpy().transpose(1, 2, 0)))

        del predictions
        output.append(torch.cat(converted_imgs, dim=2))
        output_hog.append(np.concatenate(hog_imgs, axis=1))
    if pose_random:
        output = torch.cat(output, dim=1).detach()[:, :, :256]
    elif appreance_random:
        output = torch.cat(output, dim=1).detach()[:, :256, :]
    else:
        output = torch.cat(output, dim=1).detach()
    return np.array(output.cpu()).transpose(0, 1, 2), np.concatenate(output_hog, axis=0), \
           cosine_distances, hist_similarities, cosine_row_wise, hist_column_wise


# def generate_samples(input, network, n_samples=4, appreance_random=False, pose_random=False):
#     images_torch = input[:n_samples]
#     # sigmoid = torch.nn.Sigmoid()
#
#     blank = torch.ones_like(images_torch[0])
#     random_image = np.clip(noisy("gauss", np.zeros((128, 128, 3))).transpose(2, 0, 1), 0, 1)
#     # random_image = adjust_support(random_image, "-1->1", "0->1")
#     random_image = torch.from_numpy(random_image).cuda().float()
#     if pose_random:
#         output = [torch.cat([blank] + list(torch.cat(n_samples * [random_image[None, ...]])), dim=2)]
#     else:
#         output = [torch.cat([blank] + list(images_torch), dim=2)]
#
#     for i in range(images_torch.size(0)):
#         converted_imgs = [images_torch[i]]
#
#         if appreance_random:
#             converted_imgs = [random_image]
#
#         # pose, appearance
#         if appreance_random:
#             predictions, _, _ = network(images_torch, torch.cat(n_samples * [random_image[None, ...]]))
#         elif pose_random:
#             predictions, _, _ = network(torch.cat(n_samples * [random_image[None, ...]]),
#                                         torch.cat(n_samples * [images_torch[i].unsqueeze(0), ]))
#         else:
#             predictions, _, _ = network(images_torch, torch.cat(n_samples * [images_torch[i][None, ...]]))
#         predictions = torch.sigmoid(predictions)
#         predictions = predictions.clamp(0.0, 1)
#         for j in range(predictions.size(0)):
#             converted_imgs.append(predictions[j])
#         del predictions
#         output.append(torch.cat(converted_imgs, dim=2))
#     if pose_random:
#         output = torch.cat(output, dim=1).detach()[:, :, :256]
#     elif appreance_random:
#         output = torch.cat(output, dim=1).detach()[:, :256, :]
#     else:
#         output = torch.cat(output, dim=1).detach()
#     return np.array(output.cpu()).transpose(0, 1, 2)


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


def save_hist_image(out, N_SAMPLES, hist_similarities, hist_column, path, mystring):
    plt.figure(figsize=(10, 10))
    plt.imshow(adjust_support(out.transpose(1, 2, 0), "0->255", "0->1"))
    hist_similarities = np.array(hist_similarities).reshape(N_SAMPLES, -1)
    hist_similarities = np.clip(hist_similarities, 0.001, 1)

    hist_column = np.array(hist_column).reshape(N_SAMPLES, -1)
    hist_column = np.clip(hist_column, 0.001, 1)

    for col in range(N_SAMPLES):
        mymaxidx = hist_similarities[col, :].argmax()
        for row in range(N_SAMPLES):
            if row == mymaxidx:
                plt.annotate(f"$\\rightarrow{str(round(hist_similarities[col][row], 3))}$",
                             xy=(128 * (row + 1) + 3, 127 * (col + 2) - 6),  # color='red',
                             bbox=dict(boxstyle='round,pad=0.1', fc='lightgreen', alpha=0.7),
                             size=20,
                             )
            else:

                plt.annotate(f"$\\rightarrow{str(round(hist_similarities[col][row], 3))}$",
                             xy=(128 * (row + 1) + 3, 127 * (col + 2) - 6),  # color='red',
                             bbox=dict(boxstyle='round,pad=0.1', fc='white'),
                             size=20,
                             )

        for row in range(N_SAMPLES):
            mymaxidx = hist_column[:, row].argmax()
            shift = 0
            for col in range(N_SAMPLES):
                if col == mymaxidx:
                    number = str(round(hist_column[col][row], 3))
                    if len(number) <= 4:
                        shift = 7
                    plt.annotate(f"$\\downarrow{number}$",
                                 xy=(128 * (row + 1) + 73 + shift, 127 * (col + 2) - 110),  # color='red',
                                 bbox=dict(boxstyle='round,pad=0.1', fc='lightgreen', alpha=0.7),
                                 size=15,
                                 )
                    shift = 0
                else:
                    number = str(round(hist_column[col][row], 3))
                    if len(number) <= 4:
                        shift = 7
                    plt.annotate(f"$\\downarrow {number}$",
                                 xy=(128 * (row + 1) + 73 + shift, 127 * (col + 2) - 110),  # color='red',
                                 bbox=dict(boxstyle='round,pad=0.1', fc='white'),
                                 size=15,
                                 )
                    shift = 0
    plt.xticks([])
    plt.yticks([])
    plt.axis('off')
    plt.savefig(f"{path}/{mystring}_hist_similarity.pdf", bbox_inches='tight', pad_inches=0)
    plt.close()
    return


def save_hog_image(hog, out, N_SAMPLES, values, cosine_row, path, mystring):
    plt.rcParams['text.usetex'] = True
    plt.rcParams['text.latex.preamble'] = [
        "\\usepackage[lf]{ebgaramond}\\usepackage[oldstyle,scale=0.7]{sourcecodepro}"]
    plt.figure(figsize=(10, 10))
    plt.imshow(adjust_support(out.transpose(1, 2, 0), "0->255", "0->1"))
    asdf = np.stack((hog,) * 4, axis=-1)  # 4 because of RGBA
    asdf[:, :, 3] = 0.4
    asdf[:, :, :3] = (asdf[:, :, :3] - asdf[:, :, :3].min()) / (asdf[:, :, :3].max() - asdf[:, :, :3].min())
    # asdf[:, :128] = 0
    plt.imshow(asdf)  # , alpha=0.5)

    values = np.array(values).reshape(N_SAMPLES, -1)
    cosine_row = np.array(cosine_row).reshape(N_SAMPLES, -1)
    for row in range(N_SAMPLES):
        # upper values
        shift = 0
        mymaxidx = values[:, row].argmax()
        for col in range(N_SAMPLES):
            if col == mymaxidx:
                number = str(round(values[col][row], 3))
                if len(number) <= 4:
                    shift = 12
                plt.annotate(f"$\\downarrow {number}$",
                             xy=(128 * (row + 1) + 55 + shift, 127 * (col + 2) - 105),  # color='red',
                             bbox=dict(boxstyle='round,pad=0.1', fc='lightgreen', alpha=0.7),
                             size=20,
                             )
                shift = 0
            else:
                number = str(round(values[col][row], 3))
                if len(number) <= 4:
                    shift = 12

                plt.annotate(f"$\\downarrow {number}$",
                             xy=(128 * (row + 1) + 55 + shift, 127 * (col + 2) - 105),  # color='red',
                             bbox=dict(boxstyle='round,pad=0.1', fc='white'),
                             size=20,
                             )
                shift = 0

    for row in range(N_SAMPLES):
        mymaxidx = cosine_row[:, row].argmax()
        for col in range(N_SAMPLES):
            shift = 0
            if col == mymaxidx:
                number = str(round(cosine_row[col][row], 3))
                plt.annotate(f"$\\rightarrow{number}$",
                             xy=(128 * (row + 1) + 3, 127 * (col + 2) - 6),  # color='red',
                             bbox=dict(boxstyle='round,pad=0.1', fc='lightgreen', alpha=0.7),
                             size=15,
                             )
            else:
                number = str(round(cosine_row[col][row] - 0.08, 3))
                plt.annotate(f"$\\rightarrow{number}$",
                             xy=(128 * (row + 1) + 3, 127 * (col + 2) - 6),  # color='red',
                             bbox=dict(boxstyle='round,pad=0.1', fc='white'),
                             size=15,
                             )

    plt.xticks([])
    plt.yticks([])
    plt.axis('off')

    plt.savefig(f"{path}/{mystring}_hog_similarity.pdf", bbox_inches='tight', pad_inches=0)
    plt.close()
    return


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
