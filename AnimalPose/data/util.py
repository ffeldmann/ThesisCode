import copy
from collections import namedtuple

import cv2
import numpy as np
import skimage
import skimage.transform
from skimage.draw import circle, line_aa

from AnimalPose.utils.loss_utils import heatmaps_to_coords

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


def heatmap_to_image(batch_heatmaps: np.ndarray):
    """

    Args:
        batch_heatmaps: Batch of heatmaps of shape [B, C, H, W], C == NUM_JOINTS

    Returns: Batch of images containing heatmaps of shape [B, 1, H, W]

    """
    # https://github.com/numpy/numpy/issues/9568
    np.seterr(under='ignore', invalid='ignore')
    batch, _, width, height = batch_heatmaps.shape
    images = np.sum(batch_heatmaps, axis=1).reshape(batch, 1, height, width)
    hm_min = images.min(axis=(1, 2, 3))[:, np.newaxis, np.newaxis, np.newaxis]
    hm_max = images.max(axis=(1, 2, 3))[:, np.newaxis, np.newaxis, np.newaxis]
    hm_max.clip(min=1e-6)
    images = (images - hm_min) / hm_max
    return images


def crop(image, keypoints, bbox):
    """

    Args:
        image:
        keypoints:
        bbox:

    Returns:

    """

    # bbox = np.floor(bbox).astype(int)
    img = image[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2], :]
    # First subtract the bounding box x and y from the coordinates of the keypoints
    keypoints = np.subtract(np.array(keypoints), np.array([bbox[0], bbox[1]]))
    # Set the keypoints which were zero back to zero

    keypoints[keypoints[:, 1] <= 0] = np.array([0, 0])
    return img, keypoints


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


def make_stickanimal(image, predictions):
    """
    Args:
        image: batch of images [B, W, H, C]
        joints: joint array
        predictions: batch of prediction heatmaps [B, Joints, W, H]

    Returns:

    """
    image = copy.deepcopy(image)
    if predictions.shape[-1] !=2:
        # Predictions to Keypoints
        coords, _ = heatmaps_to_coords(predictions)
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
    ]
    #  BGR color such as: Blue = a, Green = b and Red = c
    head = (255, 0, 0) # red
    body = (255, 255, 255) # white
    front = (0, 255, 0) # green
    back = (0, 0, 255) # blue

    colordict = {
        0: head,
        1: head,
        2: head,
        3: head,
        4: head,
        5: body,
        6: body,
        7: front,
        8: front,
        9: front,
        10: front,
        11: back,
        12: back,
        13: back,
        14: back,
        15: back,
    }
    for idx, img in enumerate(image):
        #import pdb; pdb.set_trace()
        img = np.zeros((img.shape[0], img.shape[1], img.shape[2]), np.uint8)
        for idx_joints, pair in enumerate(joints):
            start = coords[idx][pair[0]]
            end = coords[idx][pair[1]]
            if np.isclose(start, [0, 0]).any() or np.isclose(end, [0, 0]).any():
                continue
            cv2.line(img, (int(start[0]), int(start[1])), (int(end[0]), int(end[1])), color=colordict[idx_joints], thickness=2)
        image[idx] = img

    return image


JointModel = namedtuple(
    "JointModel",
    "body right_lines left_lines head_lines face rshoulder lshoulder headup kps_to_use total_relative_joints kp_to_joint "
    "kps_to_change kps_to_change_rel",
)

human_gait_joint_model = JointModel(
    body=[],
    right_lines=[(1, 3), (3, 5)],
    left_lines=[(0, 2), (2, 4)],
    head_lines=[(0, 1)],
    face=[],
    rshoulder=0,
    lshoulder=1,
    headup=1,
    kps_to_use=[0, 1, 2, 3, 4, 5],
    total_relative_joints=[[4, 2], [2, 0], [0, 1], [1, 3], [3, 5]],
    kp_to_joint=[
        "l_hip",
        "r_hip",
        "l_knee",
        "r_knee",
        "l_foot",
        "r_foot",
    ],
    kps_to_change=[0, 1, 2, 3, 4, 5],
    kps_to_change_rel=[0, 1, 2, 3, 4, 5],
)


def make_joint_img(
        img_shape,
        joints,
        joint_model: JointModel,
        line_colors=None,
        color_channel=None,
        scale_factor=None,
):
    # channels are opencv so g, b, r
    scale_factor = (
        img_shape[1] / scale_factor
        if scale_factor is not None
        else img_shape[1] / 128
    )
    thickness = min(int(3 * scale_factor), 1)

    imgs = list()
    for i in range(3):
        imgs.append(np.zeros(img_shape[:2], dtype="uint8"))

    if len(joint_model.body) > 2:
        body_pts = np.array([[joints[part, :] for part in joint_model.body]])
        valid_pts = np.all(np.greater_equal(body_pts, [0.0, 0.0]), axis=-1)
        if np.count_nonzero(valid_pts) > 2:
            body_pts = np.int_([body_pts[valid_pts]])
            if color_channel is None:
                body_color = (0, 127, 255)
                for i, c in enumerate(body_color):
                    cv2.fillPoly(imgs[i], body_pts, c)
            else:
                cv2.fillPoly(imgs[color_channel], body_pts, 255)

    for line_nr, line in enumerate(joint_model.right_lines):
        valid_pts = np.greater_equal(joints[line, :], [0.0, 0.0])
        if np.all(valid_pts):
            a = tuple(np.int_(joints[line[0], :]))
            b = tuple(np.int_(joints[line[1], :]))
            if color_channel is None:
                if line_colors is not None:
                    channel = int(np.nonzero(line_colors[0][line_nr])[0])
                    cv2.line(
                        imgs[channel],
                        a,
                        b,
                        color=line_colors[0][line_nr][channel],
                        thickness=thickness,
                    )
                else:
                    cv2.line(imgs[1], a, b, color=255, thickness=thickness)
            else:
                cv2.line(
                    imgs[color_channel], a, b, color=255, thickness=thickness
                )
        elif np.any(valid_pts):
            pre_p = joints[line, :]
            p = tuple(pre_p[valid_pts].astype(np.int32))

            if color_channel is None:
                cv2.circle(imgs[1], p, 5, thickness=-1, color=255)
            else:
                cv2.circle(imgs[color_channel], p, 5, thickness=-1, color=255)

    for line_nr, line in enumerate(joint_model.left_lines):

        valid_pts = np.greater_equal(joints[line, :], [0.0, 0.0])
        if np.all(valid_pts):
            a = tuple(np.int_(joints[line[0], :]))
            b = tuple(np.int_(joints[line[1], :]))
            if color_channel is None:
                if line_colors is not None:
                    channel = int(np.nonzero(line_colors[1][line_nr])[0])
                    cv2.line(
                        imgs[channel],
                        a,
                        b,
                        color=line_colors[1][line_nr][channel],
                        thickness=thickness,
                    )
                else:

                    cv2.line(imgs[0], a, b, color=255, thickness=thickness)
            else:
                cv2.line(
                    imgs[color_channel], a, b, color=255, thickness=thickness
                )
        elif np.any(valid_pts):
            pre_p = joints[line, :]
            p = tuple(pre_p[valid_pts].astype(np.int32))

            if color_channel is None:
                cv2.circle(imgs[0], p, 5, thickness=-1, color=255)
            else:
                cv2.circle(imgs[color_channel], p, 5, thickness=-1, color=255)

    if len(joint_model.head_lines) == 0:
        rs = joints[joint_model.rshoulder, :]
        ls = joints[joint_model.lshoulder, :]
        cn = joints[joint_model.headup, :]
        if np.any(np.less(np.stack([rs, ls], axis=-1), [0.0, 0.0])):
            neck = np.asarray([-1.0, -1.0])
        else:
            neck = 0.5 * (rs + ls)

        pts = np.stack([neck, cn], axis=-1).transpose()

        valid_pts = np.greater_equal(pts, [0.0, 0.0])
        throat_len = np.asarray([0], dtype=np.float)
        if np.all(valid_pts):
            throat_len = np.linalg.norm(pts[0] - pts[1])
            if color_channel is None:
                a = tuple(np.int_(pts[0, :]))
                b = tuple(np.int_(pts[1, :]))
                cv2.line(imgs[0], a, b, color=127, thickness=thickness)
                cv2.line(imgs[1], a, b, color=127, thickness=thickness)
            else:
                cv2.line(
                    imgs[color_channel],
                    tuple(np.int_(pts[0, :])),
                    tuple(np.int_(pts[1, :])),
                    color=255,
                    thickness=thickness,
                )
        elif np.any(valid_pts):
            throat_len = np.asarray([0], dtype=np.float)
            p = tuple(
                pts[np.all(valid_pts, axis=-1), :].astype(np.int32).squeeze()
            )
            if color_channel is None:
                cv2.circle(imgs[0], p, 5, color=127, thickness=-1)
                cv2.circle(imgs[1], p, 5, color=127, thickness=-1)
            else:
                cv2.circle(imgs[color_channel], p, 5, color=255, thickness=-1)
    else:
        throat_lens = np.zeros(len(joint_model.head_lines), dtype=np.float)
        for line_nr, line in enumerate(joint_model.head_lines):

            valid_pts = np.greater_equal(joints[line, :], [0.0, 0.0])
            if np.all(valid_pts):
                throat_lens[line_nr] = np.linalg.norm(
                    joints[line[0], :] - joints[line[1], :]
                )
                a = tuple(np.int_(joints[line[0], :]))
                b = tuple(np.int_(joints[line[1], :]))
                if color_channel is None:
                    if line_colors is not None:
                        channel = int(np.nonzero(line_colors[2][line_nr])[0])
                        cv2.line(
                            imgs[channel],
                            a,
                            b,
                            color=line_colors[2][line_nr][channel],
                            thickness=thickness,
                        )
                    else:
                        cv2.line(imgs[0], a, b, color=127, thickness=thickness)
                        cv2.line(imgs[1], a, b, color=127, thickness=thickness)
                else:
                    cv2.line(
                        imgs[color_channel],
                        a,
                        b,
                        color=255,
                        thickness=thickness,
                    )
            elif np.any(valid_pts):
                pre_p = joints[line, :]
                p = tuple(pre_p[valid_pts].astype(np.int32))

                if color_channel is None:
                    cv2.circle(imgs[0], p, 5, thickness=-1, color=255)
                else:
                    cv2.circle(
                        imgs[color_channel], p, 5, thickness=-1, color=255
                    )

        throat_len = np.amax(throat_lens)

    if len(joint_model.face) > 0:
        for line_nr, line in enumerate(joint_model.face):

            valid_pts = np.greater_equal(joints[line, :], [0.0, 0.0])
            if np.all(valid_pts):
                if (
                        np.linalg.norm(joints[line[0], :] - joints[line[1], :])
                        < throat_len
                ):
                    a = tuple(np.int_(joints[line[0], :]))
                    b = tuple(np.int_(joints[line[1], :]))
                    if color_channel is None:
                        if line_colors is not None:
                            channel = int(
                                np.nonzero(line_colors[2][line_nr])[0]
                            )
                            cv2.line(
                                imgs[channel],
                                a,
                                b,
                                color=line_colors[2][line_nr][channel],
                                thickness=thickness,
                            )
                        else:
                            cv2.line(
                                imgs[0], a, b, color=127, thickness=thickness
                            )
                            cv2.line(
                                imgs[1], a, b, color=127, thickness=thickness
                            )
                    else:
                        cv2.line(
                            imgs[color_channel],
                            a,
                            b,
                            color=255,
                            thickness=thickness,
                        )
            elif np.any(valid_pts):
                pre_p = joints[line, :]
                p = tuple(pre_p[valid_pts].astype(np.int32))

                if color_channel is None:
                    cv2.circle(imgs[0], p, 5, thickness=-1, color=255)
                else:
                    cv2.circle(
                        imgs[color_channel], p, 5, thickness=-1, color=255
                    )

    img = np.stack(imgs, axis=-1)
    if img_shape[-1] == 1:
        img = np.mean(img, axis=-1)[:, :, None]
    # img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    return img


def add_joints_to_img(img, kps, joints, color_kps=[[255, 0, 0]], color_joints=[[255, 0, 0]]):
    # params
    border_safety = 25
    h, w = img.shape[0:2]
    r_1 = int(h / 250)

    # draw keypoints
    if len(color_kps) == 1:
        color_kps = [color_kps[0] for _ in range(kps.shape[0])]

    for i, kp in enumerate(kps):
        x = np.min([w - border_safety, kp[0]])  # x
        y = np.min([h - border_safety, kp[1]])  # y
        rr, cc = circle(y, x, r_1)
        img[rr, cc, 0] = color_kps[i][0]
        img[rr, cc, 1] = color_kps[i][1]
        img[rr, cc, 2] = color_kps[i][2]

    # draw joints
    if len(color_joints) == 1:
        color_joints = [color_joints[0] for _ in range(len(joints))]

    for i, jo in enumerate(joints):
        rr, cc, val = line_aa(int(kps[jo[0], 1]), int(kps[jo[0], 0]), int(kps[jo[1], 1]),
                              int(kps[jo[1], 0]))  # [jo_0_y, jo_0_x, jo_1_y, jo_1_x]

        img[rr, cc, 0] = color_joints[i][0] * val
        img[rr, cc, 1] = color_joints[i][1] * val
        img[rr, cc, 2] = color_joints[i][2] * val

    return img
