import torch
import cv2
import numpy as np
import skimage
import skimage.transform
from skimage.draw import circle, line, line_aa

#JointModel: Type[JointModel] = namedtuple("JointModel","body right_lines left_lines rshoulder lshoulder headup kps_to_use kp_to_joint")

class Rescale(object):
    """Rescale the image in a sample to a given size.

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
        #bbox = bbox * [new_w / w, new_h / h, new_w / w, new_h / h]

        return img, keypoints

def gaussian_k(x0, y0, sigma, width, height):
    """ Make a square gaussian kernel centered at (x0, y0) with sigma as SD.
    """
    x = np.arange(0, width, 1, float)  ## (width,)
    y = np.arange(0, height, 1, float)[:, np.newaxis]  ## (height,1)
    return np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

def make_heatmaps(image, keypoints, sigma=0.5):
    hm = np.zeros((len(keypoints), image.shape[0], image.shape[1]),  dtype=np.float32)
    for idx in range(0, len(keypoints)):
        if not np.array_equal(keypoints[idx], [0, 0]):

            hm[idx,:, :] = gaussian_k(keypoints[idx][0],
                                        keypoints[idx][1],
                                        sigma, image.shape[1], image.shape[0])
        else:
            hm[idx, :, :] = np.zeros((image.shape[0], image.shape[1]))  # height, width
    return hm.reshape(-1, image.shape[0], image.shape[1])

def make_stickanimal(image, keypoints, sigma=0.8):
    hm = np.zeros((len(keypoints), image.shape[0], image.shape[1]),  dtype=np.float32)
    for idx in range(0, len(keypoints)):
        if not np.array_equal(keypoints[idx], [0, 0]):

            hm[idx,:, :] = gaussian_k(keypoints[idx][0],
                                        keypoints[idx][1],
                                        sigma, image.shape[1], image.shape[0])
        else:
            hm[idx, :, :] = np.zeros((image.shape[0], image.shape[1]))  # height, width
    return hm

def make_joint_img(img_shape, joints,JointModel, color_channel=None):
    # channels are opencv so g, b, r
    scale_factor = img_shape[1] / 128
    thickness = int(3 * scale_factor)
    imgs = list()
    for i in range(3):
        imgs.append(np.zeros(img_shape[:2], dtype="uint8"))

    body_pts = np.array([[joints[part, :] for part in joint_model.body]])
    valid_pts = np.all(np.greater_equal(body_pts, [0.0, 0.0]), axis=-1)
    if np.count_nonzero(valid_pts) > 2:
        body_pts = np.int_([body_pts[valid_pts]])
        if color_channel is None:
            cv2.fillPoly(imgs[2], body_pts, 255)
        else:
            cv2.fillPoly(imgs[color_channel], body_pts, 255)

    for line in joint_model.right_lines:
        valid_pts = np.greater_equal(joints[line, :], [0.0, 0.0])
        if np.all(valid_pts):
            a = tuple(np.int_(joints[line[0], :]))
            b = tuple(np.int_(joints[line[1], :]))
            if color_channel is None:
                cv2.line(imgs[1], a, b, color=255, thickness=thickness)
            else:
                cv2.line(
                    imgs[color_channel], a, b, color=255, thickness=thickness
                )
        elif np.any(valid_pts):
            pre_p = joints[line, :]
            p = tuple(
                pre_p[valid_pts]
                .astype(np.int32)
            )

            if color_channel is None:
                cv2.circle(imgs[1], p, 5, thickness=-1, color=255)
            else:
                cv2.circle(imgs[color_channel], p, 5, thickness=-1, color=255)

    for line in joint_model.left_lines:

        valid_pts = np.greater_equal(joints[line, :], [0.0, 0.0])
        if np.all(valid_pts):
            a = tuple(np.int_(joints[line[0], :]))
            b = tuple(np.int_(joints[line[1], :]))
            if color_channel is None:
                cv2.line(imgs[0], a, b, color=255, thickness=thickness)
            else:
                cv2.line(
                    imgs[color_channel], a, b, color=255, thickness=thickness
                )
        elif np.any(valid_pts):
            pre_p = joints[line,:]
            p = tuple(
                pre_p[valid_pts]
                    .astype(np.int32)
            )

            if color_channel is None:
                cv2.circle(imgs[0], p, 5, thickness=-1, color=255)
            else:
                cv2.circle(imgs[color_channel], p, 5, thickness=-1, color=255)

    rs = joints[joint_model.rshoulder, :]
    ls = joints[joint_model.lshoulder, :]
    cn = joints[joint_model.headup, :]
    if np.any(np.less(np.stack([rs, ls], axis=-1), [0.0, 0.0])):
        neck = np.asarray([-1.0, -1.0])
    else:
        neck = 0.5 * (rs + ls)

    pts = np.stack([neck, cn], axis=-1).transpose()

    valid_pts = np.greater_equal(pts, [0.0, 0.0])
    if np.all(valid_pts):
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
        p = tuple(pts[np.all(valid_pts, axis=-1), :].astype(np.int32).squeeze())
        if color_channel is None:
            cv2.circle(imgs[0], p, 5, color=127, thickness=-1)
            cv2.circle(imgs[1], p, 5, color=127, thickness=-1)
        else:
            cv2.circle(imgs[color_channel], p, 5, color=255, thickness=-1)

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