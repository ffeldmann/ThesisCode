import imgaug.augmenters as iaa
import numpy as np
import skimage.color
from edflow.data.agnostics.subdataset import SubDataset
from edflow.data.believers.meta import MetaDataset
from edflow.data.dataset_mixin import DatasetMixin
from edflow.data.believers.sequence import SequenceDataset
import sklearn.model_selection
from AnimalPose.data.util import make_heatmaps, Rescale, crop
from edflow.data.util import adjust_support
from edflow import get_logger


class MPII_Sequence(MetaDataset):
    def __init__(self, config):
        super().__init__(config["dataroot"])
        self.config = config
        self.crop = crop


class MPII_Sequence_Abstract(DatasetMixin):
    def __init__(self, config, mode="all"):
        assert mode in ["train", "validation", "all"], f"Should be train, validation or all, got {mode}"
        self.config = config
        self.sequence_length = 2  # if config.get("sequence_length", False) == False else config["sequence_length"]
        # self.sc = Animal_Sequence(config)
        self.sc = MPII_Sequence(config)
        # works if dataroot like "VOC2011/cats_meta"
        # TODO PROBABLY NOT CORRECT HERE
        self.animal = config["dataroot"].split("/")[1].split("_")[0]

        self.train = int(config["train_size"] * len(self.sc))
        self.test = 1 - self.train
        self.sigma = config["sigma"]
        self.augmentation = config["augmentation"]
        self.logger = get_logger(self)

        self.resize = iaa.Resize(self.config["resize_to"])
        self.aug_factor = 0.5
        self.seq = iaa.Sequential([
            iaa.Sometimes(self.aug_factor + 0.2, iaa.Fliplr()),
            iaa.Sometimes(self.aug_factor, iaa.Flipud()),
        ], random_order=True)

        if mode != "all":
            # split_indices = np.arange(self.train) if mode == "train" else np.arange(self.train + 1, len(self.sc))
            dset_indices = np.arange(len(self.sc))
            train_indices, test_indices = sklearn.model_selection.train_test_split(dset_indices,
                                                                                   train_size=float(
                                                                                       config["train_size"]),
                                                                                   random_state=int(
                                                                                       config["random_state"]))
            if mode == "train":
                self.data = SubDataset(self.sc, train_indices)
            else:
                self.data = SubDataset(self.sc, test_indices)
        else:
            self.data = self.sc

    def get_appearance_image(self, videoid, frameid):
        """

        Args:
            videoid: video id
            frameid: frame id

        Returns:

        """
        # Get frame id mask where
        fid_mask = np.invert(np.isin(self.data.labels["fid"], frameid))
        try:
            valid_indices = np.argwhere((videoid == self.data.labels["global_video_class"]) & fid_mask).squeeze(-1)
            if valid_indices.size == 0:
                valid_indices = np.argwhere((videoid == self.data.labels["global_video_class"])).squeeze(-1)
        except IndexError:
            valid_indices = np.argwhere((videoid == self.data.labels["global_video_class"])).squeeze(-1)
        random_idx = np.random.choice(valid_indices)

        if videoid != self.data[random_idx]["labels_"]["global_video_class"]:
            import pdb;
            pdb.set_trace()
        return self.data[random_idx]

    def get_example(self, idx: object) -> object:
        """
        Args:
            idx: integer indicating index of dataset

        Returns: example element from dataset

        """
        example = super().get_example(idx)
        output = dict()
        output["global_video_class0"] = example["labels_"]["global_video_class"]
        output["fid0"] = example["labels_"]["fid"]
        appearance_example = self.get_appearance_image(output["global_video_class0"], output["fid0"])
        output["global_video_class1"] = appearance_example["labels_"]["global_video_class"]
        if output["global_video_class0"] != output["global_video_class1"]:
            import pdb;
            pdb.set_trace()
        output["fid1"] = appearance_example["labels_"]["fid"]

        for i, ex in enumerate([example, appearance_example]):
            if self.config.get("image_type", "") == "mask":
                image = ex["masked_frames"]()
                output[f"framename{i}"] = ex["labels_"]["masked_frames_"]
            elif self.config.get("image_type", "") == "white":
                image = ex["whitened_frames"]()
                output[f"framename{i}"] = ex["labels_"]["whitened_frames_"]
            else:
                image = ex["frames"]()
                output[f"framename{i}"] = ex["labels_"]["frames_"]
            try:
                keypoints = ex["kps"]
            except:
                keypoints = np.array(
                    [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0],
                     [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]])
            try:
                bboxes = ex["bboxes"]
                bbox_available = True
            except:
                # self.logger.warning("No bboxes in this dataset!")
                bbox_available = False

            # store which keypoints are not present in the dataset
            zero_mask_x = np.where(keypoints[:, 0] <= 0)
            zero_mask_y = np.where(keypoints[:, 1] <= 0)
            # need uint 8 for augmentation methods
            image = adjust_support(image, "0->255")
            if "crop" in self.config.keys():
                if self.config["crop"]:
                    if not bbox_available:
                        pass
                        # self.logger.warning("No resizing possible, no bounding box!")
                    else:
                        image, keypoints = crop(image, keypoints, bboxes)
            if self.augmentation and i == 1:
                # randomly perform some augmentations on the image, keypoints and bboxes
                image, keypoints = self.seq(image=image, keypoints=keypoints.reshape(1, -1, 2))
            # (H, W, C) and keypoints need to be reshaped from (N,J,2) -> (J,2)  J==Number of joints / keypoint pairs
            image, keypoints = self.resize(image=image, keypoints=keypoints.reshape(1, -1, 2))
            # image, keypoints = self.rescale(image, keypoints.reshape(-1, 2))
            keypoints = keypoints.reshape(-1, 2)
            keypoints[zero_mask_x] = np.array([0, 0])
            keypoints[zero_mask_y] = np.array([0, 0])
            # we always work with "0->1" images and np.float32
            height = image.shape[0]
            width = image.shape[1]

            output[f"inp{i}"] = adjust_support(image, "0->1")
            output[f"kps{i}"] = keypoints
            output[f"targets{i}"] = adjust_support(make_heatmaps(output[f"inp{i}"], keypoints, sigma=self.sigma),
                                                   "0->1")
        assert output["global_video_class0"] == output[
            "global_video_class1"], f"Video classes need to be the same! Got {output['global_video_class0']} and {output['global_video_class1']}"
        return output


class MPII_Sequence_AbstractDONOTUSE(DatasetMixin):
    def __init__(self, config, mode="all"):
        assert mode in ["train", "validation", "all"], f"Should be train, validation or all, got {mode}"
        self.config = config
        self.sequence_length = 30
        self.sc = SequenceDataset(MPII_Sequence(config), self.sequence_length, step=config["sequence_step_size"])
        self.train = int(config["train_size"] * len(self.sc))
        self.test = 1 - self.train
        self.sigma = config["sigma"]
        self.augmentation = config["augmentation"]
        self.aug_factor = 0.5
        self.resize = iaa.Resize(self.config["resize_to"])
        if self.augmentation:
            self.seq = iaa.Sequential([
                # iaa.Sometimes(self.aug_factor, iaa.AdditiveGaussianNoise(scale=0.05 * 255)),
                # iaa.Sometimes(self.aug_factor, iaa.SaltAndPepper(0.01, per_channel=False)),
                # iaa.Sometimes(self.aug_factor, iaa.CoarseDropout(0.01, size_percent=0.5)),
                iaa.Fliplr(self.aug_factor),
                iaa.Flipud(self.aug_factor),
                # iaa.Sometimes(self.aug_factor, iaa.GaussianBlur(sigma=(0, 3.0))),
                # iaa.LinearContrast((0.75, 1.5)),
                # Convert each image to grayscale and then overlay the
                # result with the original with random alpha. I.e. remove
                # colors with varying strengths.
                # iaa.Grayscale(alpha=(0.0, 1.0)),
            ], random_order=True)

        if mode != "all":
            # split_indices = np.arange(self.train) if mode == "train" else np.arange(self.train + 1, len(self.sc))
            dset_indices = np.arange(len(self.sc))
            train_indices, test_indices = sklearn.model_selection.train_test_split(dset_indices,
                                                                                   train_size=float(
                                                                                       config["train_size"]),
                                                                                   random_state=int(
                                                                                       config["random_state"]))
            if mode == "train":
                self.data = SubDataset(self.sc, train_indices)
            else:
                self.data = SubDataset(self.sc, test_indices)
        else:
            self.data = self.sc

    def get_appearance_image(self, pid, fids):
        """

        Args:
            pid: video id
            fids: frame ids

        Returns:

        """
        # Get frame id mask where
        fid_mask = np.invert(np.isin(self.data.labels["fid"], fids))
        try:
            valid_indices = np.argwhere((pid == self.data.labels["pid"]) & fid_mask)[0]
        except IndexError:
            valid_indices = np.argwhere((pid == self.data.labels["pid"]))[0]
        random_idx = np.random.choice(valid_indices)
        return self.data[random_idx]

    def get_example(self, idx: object) -> object:
        """
        Args:
            idx: integer indicating index of dataset

        Returns: example element from dataset

        """
        example = super().get_example(idx)
        output = dict()
        import pdb;
        pdb.set_trace()
        sample_idxs = np.random.choice(np.arange(0, self.sequence_length), 2, replace=False)
        for i, ex_idx in enumerate(sample_idxs):
            if self.config.get("image_type", "") == "mask":
                image = example["masked_frames"][ex_idx]()
            elif self.config.get("image_type", "") == "white":
                image = example["whitened_frames"][ex_idx]()
            else:
                image = example["frames"][ex_idx]()
            try:
                keypoints = self.labels["kps"][idx][ex_idx]
            except:
                keypoints = np.array(
                    [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0],
                     [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]])
            try:
                bboxes = self.labels["bboxes"][idx][ex_idx]
                bbox_available = True
            except:
                # self.logger.warning("No bboxes in this dataset!")
                bbox_available = False
            output[f"fid{i}"] = self.labels["fid"][idx][ex_idx]
            # store which keypoints are not present in the dataset
            zero_mask_x = np.where(keypoints[:, 0] <= 0)
            zero_mask_y = np.where(keypoints[:, 1] <= 0)
            # need uint 8 for augmentation methods
            image = adjust_support(image, "0->255")
            if "crop" in self.config.keys():
                if self.config["crop"]:
                    if not bbox_available:
                        pass
                        # self.logger.warning("No resizing possible, no bounding box!")
                    else:
                        image, keypoints = crop(image, keypoints, bboxes)
            if self.augmentation and i == 1:
                # randomly perform some augmentations on the image, keypoints and bboxes
                image, keypoints = self.seq(image=image, keypoints=keypoints.reshape(1, -1, 2))
            # (H, W, C) and keypoints need to be reshaped from (N,J,2) -> (J,2)  J==Number of joints / keypoint pairs
            image, keypoints = self.resize(image=image, keypoints=keypoints.reshape(1, -1, 2))
            # image, keypoints = self.rescale(image, keypoints.reshape(-1, 2))
            keypoints = keypoints.reshape(-1, 2)
            keypoints[zero_mask_x] = np.array([0, 0])
            keypoints[zero_mask_y] = np.array([0, 0])
            # we always work with "0->1" images and np.float32
            # image = adjust_support(image, "0->1")
            height = image.shape[0]
            width = image.shape[1]
            if "as_grey" in self.config.keys():
                if self.config["as_grey"]:
                    output[f"inp{i}"] = adjust_support(skimage.color.rgb2gray(image).reshape(height, width, 1), "0->1")
                    assert (self.data.data.config["n_channels"] == 1), (
                        "n_channels should be 1, got {}".format(self.config["n_channels"]))
                else:
                    output[f"inp{i}"] = adjust_support(image, "0->1")
            else:
                output[f"inp{i}"] = adjust_support(image, "0->1")

            output[f"targets{i}"] = adjust_support(make_heatmaps(output[f"inp{i}"], keypoints, sigma=self.sigma),
                                                   "0->1")
            output[f"framename{i}"] = self.labels["frames_"][idx][ex_idx]
        return output


class MPII_Sequence_Train(MPII_Sequence_Abstract):
    def __init__(self, config):
        super().__init__(config, mode="train")


class MPII_Sequence_Validation(MPII_Sequence_Abstract):
    def __init__(self, config):
        super().__init__(config, mode="validation")
        self.augmentation = False


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import sys


    def info(type, value, tb):
        if hasattr(sys, 'ps1') or not sys.stderr.isatty():
            # we are in interactive mode or we don't have a tty-like
            # device, so we call the default hook
            sys.__excepthook__(type, value, tb)
        else:
            import traceback, pdb
            # we are NOT in interactive mode, print the exception...
            traceback.print_exception(type, value, tb)
            print
            # ...then start the debugger in post-mortem mode.
            pdb.pm()


    sys.excepthook = info

    # DATAROOT = {"dataroot": '/export/home/ffeldman/Masterarbeit/data/VOC2011/cats_meta'}
    # cats = CatsVOC2011(DATAROOT)
    # ex = cats.get_example(3)
    # for hm in ex["targets"]:
    #     print(hm.shape)
    #     plt.imshow(hm)
    #     plt.show()
