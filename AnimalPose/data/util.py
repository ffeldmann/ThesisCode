import cv2
import numpy as np
import skimage
import skimage.transform
from edflow.data.util import adjust_support

from AnimalPose.utils.image_utils import heatmaps_to_coords
from AnimalPose.utils.tensor_utils import torch2numpy


def bboxes_from_kps(keypoints_raw):
    # from jhaux flowtrack_j
    '''Findes a bounding box of all sets of keypoints in ``keypoints``.

    Arguments:
        keypoints (np.array): Nd-array, where the last 2 dimensions correspond
            to the ``N`` keypoints of one instance. Shape: ``[..., N, 2]``

    Returns:
        np.array: Boxes around all instances. Shape: ``[..., 4]``.
    '''
    # fix treating of 0,0 point
    mask = np.not_equal(keypoints_raw[:, 0], 0) * np.not_equal(keypoints_raw[:, 1], 0)
    keypoints = keypoints_raw[mask]
    if not len(keypoints):
        return np.array([0, 0, 0, 0])
    # keypoints= keypoints_raw

    x_mins = np.amin(keypoints[..., 0], axis=-1, keepdims=True)  # [..., xmin]
    y_mins = np.amin(keypoints[..., 1], axis=-1, keepdims=True)  # [..., ymin]
    kp_mins = np.concatenate([x_mins, y_mins], -1)  # [..., xmin ymin]

    x_maxs = np.amax(keypoints[..., 0], axis=-1, keepdims=True)  # [..., xmax]
    y_maxs = np.amax(keypoints[..., 1], axis=-1, keepdims=True)  # [..., ymax]
    kp_maxs = np.concatenate([x_maxs, y_maxs], -1)  # [..., xmax ymax]

    widths_and_heights = kp_maxs - kp_mins  # [..., width height]

    # [..., xmin ymin width height]
    return np.concatenate([kp_mins, widths_and_heights], axis=-1)


class Rescale(object):
    """Rescale the image and keypoints in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, image, keypoints):

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = skimage.transform.resize(image, (new_h, new_w))
        # h and w are swapped for keypoints because for images,
        # x and y axes are axis 1 and 0 respectively
        keypoints = keypoints * [new_w / w, new_h / h]
        # bbox = bbox * [new_w / w, new_h / h, new_w / w, new_h / h]

        return img, keypoints


def crop(image, keypoints, bbox):
    """

    Args:
        image:
        keypoints:
        bbox:

    Returns:

    """

    # bbox = np.floor(bbox).astype(int)
    prev_width, prev_height, _ = image.shape
    img = image[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2], :]

    zero_mask_x = np.where(keypoints[:, 0] <= 0)
    zero_mask_y = np.where(keypoints[:, 1] <= 0)
    # First subtract the bounding box x and y from the coordinates of the keypoints
    keypoints = np.subtract(np.array(keypoints), np.array([bbox[0], bbox[1]]))
    # Set the keypoints which were zero back to zero
    # keypoints[keypoints[:, 1] <= 0] = np.array([0, 0])

    keypoints[zero_mask_x] = np.array([0, 0])
    keypoints[zero_mask_y] = np.array([0, 0])

    return img, keypoints.astype(np.float32)


def gaussian_k(x0, y0, sigma, height, width):
    """ Make a square gaussian kernel centered at (x0, y0) with sigma as SD.
    """
    x = np.arange(0, width, 1, float)  ## (width,)
    y = np.arange(0, height, 1, float)[:, np.newaxis]  ## (height,1)
    return np.exp(-((x - int(x0)) ** 2 + (y - int(y0)) ** 2) / (2 * sigma ** 2))


def make_heatmaps(image, keypoints, sigma=0.5):
    """

    Args:
        image:
        keypoints:
        sigma:

    Returns:

    """
    hm = np.zeros((len(keypoints), image.shape[0], image.shape[1]), dtype=np.float32)
    for idx in range(0, len(keypoints)):
        if not np.array_equal(keypoints[idx], [0, 0]):

            hm[idx, :, :] = gaussian_k(keypoints[idx][0],
                                       keypoints[idx][1],
                                       sigma, image.shape[0], image.shape[1])
        else:
            hm[idx, :, :] = np.zeros((image.shape[0], image.shape[1]))  # height, width
    return hm


def make_stickanimal(image, predictions, thresh=0, draw_all_circles=True):
    """
    Args:
        image: batch of images [B, W, H, C]
        joints: joint array
        predictions: batch of prediction heatmaps [B, Joints, W, H]

    Returns:

    """
    image = adjust_support(np.copy(image), "0->255").astype(np.uint8)
    if predictions.shape[-1] != 2:
        # Predictions to Keypoints
        coords, _ = heatmaps_to_coords(torch2numpy(predictions), thresh=thresh)
    else:
        coords = predictions

    joints = [
        # Head
        [2, 0],  # Nose - L_Eye
        [2, 1],  # Nose - R_Eye
        [0, 3],  # L_Eye - L_EarBase
        [1, 4],  # R_Eye - R_EarBase
        [2, 8],  # Nose - Throat
        # Body
        [8, 9],  # Throat - L_F_Elbow
        [8, 5],  # Throat - R_F_Elbow
        [9, 10],  # L_F_Elbow - Withers
        [5, 10],  # R_F_Elbow - Withers
        # Front
        [9, 16],  # L_F_Elbow - L_F_Knee
        [16, 6],  # L_F_Knee - L_F_Paw
        [5, 17],  # R_F_Elbow - R_F_Knee
        [17, 7],  # R_F_Knee - R_F_Paw
        # Back
        [14, 18],  # L_B_Elbow - L_B_Knee
        [18, 12],  # L_B_Knee - L_B_Paw
        [15, 19],  # R_B_Elbow - R_B_Knee
        [19, 13],  # R_B_Knee - R_B_Paw
        [10, 11],  # Withers - TailBase
        [11, 15],  # Tailbase - R_B_Elbow
        [11, 14],  # Tailbase - L_B_Elbow
    ]
    #  BGR color such as: Blue = a, Green = b and Red = c
    head = (255, 0, 0)  # red
    body = (255, 255, 255)  # white
    front = (0, 255, 0)  # green
    back = (0, 0, 255)  # blue

    colordict = {
        0: head,
        1: head,
        2: head,
        3: head,
        4: head,
        5: body,
        6: body,
        7: body,
        8: body,
        9: front,
        10: front,
        11: front,
        12: front,
        13: back,
        14: back,
        15: back,
        16: back,
        17: back,
        18: back,
        19: back,
    }
    for idx, orig in enumerate(image):
        img = np.zeros((orig.shape[0], orig.shape[1], orig.shape[2]), np.uint8)
        img[:, :, :] = orig  # but why not img = orig.copy() ?
        for idx_joints, pair in enumerate(joints):
            start = coords[idx][pair[0]]
            end = coords[idx][pair[1]]

            # catch the case, that both points are missing (are 0,0) or we want to draw the circles
            if np.isclose(start, [0, 0]).any() and np.isclose(end, [0, 0]).any():
                continue
            # catch the case, that only one of them is missing
            if not np.isclose(start, [0, 0]).any() and draw_all_circles:
                cv2.circle(img, (int(start[0]), int(start[1])), radius=1,
                           color=colordict[idx_joints], thickness=2, lineType=cv2.LINE_AA)
            if not np.isclose(end, [0, 0]).any() and draw_all_circles:
                cv2.circle(img, (int(end[0]), int(end[1])), radius=1,
                           color=colordict[idx_joints], thickness=2, lineType=cv2.LINE_AA)

            if not np.isclose(start[0], 0):
                if not np.isclose(end[0], 0):
                    cv2.line(img, (int(start[0]), int(start[1])), (int(end[0]), int(end[1])),
                             color=colordict[idx_joints], thickness=1, lineType=cv2.LINE_AA)
        image[idx] = img

    return adjust_support(image, "-1->1")
