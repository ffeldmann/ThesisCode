import imgaug.augmenters as iaa
import numpy as np
import skimage.color
from edflow.data.agnostics.subdataset import SubDataset
from edflow.data.believers.meta import MetaDataset
from edflow.data.dataset_mixin import DatasetMixin
import sklearn.model_selection
from AnimalPose.data.util import make_heatmaps, crop, make_stickanimal
from AnimalPose.utils.image_utils import heatmaps_to_image

from AnimalPose.data.util import bboxes_from_kps

from edflow.data.util import adjust_support
from edflow import get_logger

animal_class = {"cats": 0,
                "dogs": 1,
                "sheeps": 2,
                "cows": 3,
                "horses": 4,
                "tigers": 5,
                "domestic": 2,  # also sheeps
                "hellenic": 1,  # also a dog
                }

parts = {
    "L_Eye": 0,
    "R_Eye": 1,
    "Nose": 2,
    "L_EarBase": 3,
    "R_EarBase": 4,
    "R_F_Elbow": 5,
    "L_F_Paw": 6,
    "R_F_Paw": 7,
    "Throat": 8,
    "L_F_Elbow": 9,
    "Withers": 10,
    "TailBase": 11,
    "L_B_Paw": 12,
    "R_B_Paw": 13,
    "L_B_Elbow": 14,
    "R_B_Elbow": 15,
    "L_F_Knee": 16,
    "R_F_Knee": 17,
    "L_B_Knee": 18,
    "R_B_Knee": 19,
}

idx_to_part = {v: k for k, v in parts.items()}


class AnimalTriplet(MetaDataset):
    def __init__(self, config):
        super().__init__(config["dataroot"])
        self.config = config
        self.crop = crop
        self.logger = get_logger(self)
        self.animal = config["dataroot"].split("/")[1].split("_")[0]


class AnimalTriplet_Abstract(DatasetMixin):
    def __init__(self, config, mode="all"):
        assert mode in ["train", "validation", "all"], f"Should be train, validation or all, got {mode}"
        self.config = config
        self.sc = AnimalTriplet(config)
        self.train = int(config["train_size"] * len(self.sc))
        self.test = 1 - self.train
        self.augmentation = config["augmentation"]
        self.aug_factor = 0.5
        self.resize = iaa.Resize(self.config["resize_to"])
        if self.augmentation:
            self.seq = iaa.Sequential([
                iaa.Sometimes(self.aug_factor, iaa.SaltAndPepper(0.01, per_channel=False)),
                iaa.Sometimes(self.aug_factor, iaa.CoarseDropout(0.01, size_percent=0.5)),
                iaa.Fliplr(self.aug_factor),
                iaa.Flipud(self.aug_factor),
                iaa.Sometimes(self.aug_factor, iaa.GaussianBlur(sigma=(0, 3.0))),
                iaa.LinearContrast((0.75, 1.5)),
                # Convert each image to grayscale and then overlay the
                # result with the original with random alpha. I.e. remove
                # colors with varying strengths.
                iaa.Grayscale(alpha=(0.0, 1.0)),
            ], random_order=True)

        self.joints = [
            [2, 0],  # Nose - L_Eye
            [2, 1],  # Nose - R_Eye
            [0, 3],  # L_Eye - L_EarBase
            [1, 4],  # R_Eye - R_EarBase
            [2, 8],  # Nose - Throat
            [8, 9],  # Throat - L_F_Elbow
            [8, 5],  # Throat - R_F_Elbow
            [9, 16],  # L_F_Elbow - L_F_Knee
            [16, 6],  # L_F_Knee - L_F_Paw
            [5, 17],  # R_F_Elbow - R_F_Knee
            [17, 7],  # R_F_Knee - R_F_Paw
            [14, 18],  # L_B_Elbow - L_B_Knee
            [18, 13],  # L_B_Knee - L_B_Paw
            [15, 19],  # R_B_Elbow - R_B_Knee
            [19, 13],  # R_B_Knee - R_B_Paw
            [10, 11],  # Withers - TailBase
        ]

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

    def get_parts(self):
        return parts

    def get_idx_parts(self, idx):
        reverse_list = {v: k for k, v in parts.items()}
        return reverse_list[idx]

    def get_example(self, idx: object) -> object:
        """
        Args:
            idx: integer indicating index of dataset

        Returns: example element from dataset

        """
        example = super().get_example(idx)
        # Images are loaded with support from 0->255
        output = {}
        if self.config.get("image_type", "") == "mask":
            image_p0a0 = example["p0a0_masked_frames"]()
            image_p0a1 = example["p0a1_masked_frames"]()
            image_p1a1 = example["p1a1_masked_frames"]()
        elif self.config.get("image_type", "") == "white":
            image_p0a0 = example["p0a0_whitened_frames"]()
            image_p0a1 = example["p0a1_whitened_frames"]()
            image_p1a1 = example["p1a1_whitened_frames"]()
        else:
            image_p0a0 = example["p0a0_frames"]()
            image_p0a1 = example["p0a1_frames"]()
            image_p1a1 = example["p1a1_frames"]()

        if self.augmentation:
            # randomly perform some augmentations on the image, keypoints and bboxes
            image_p0a0 = self.seq(image=image_p0a0)
            image_p0a1 = self.seq(image=image_p0a1)
            image_p1a1 = self.seq(image=image_p1a1)

        image_p0a0 = self.resize(image=image_p0a0)
        image_p0a1 = self.resize(image=image_p0a1)
        image_p1a1 = self.resize(image=image_p1a1)

        output["inp0"] = adjust_support(image_p0a0, "0->1")  # p0a0
        output["inp1"] = adjust_support(image_p0a1, "0->1")  # p0a1
        output["inp2"] = adjust_support(image_p1a1, "0->1")  # p1a1
        output["animal_class"] = np.array(animal_class[self.data.data.animal])
        return output


class AnimalTriplet_Train(AnimalTriplet_Abstract):
    def __init__(self, config):
        super().__init__(config, mode="train")


class AnimalTriplet_Validation(AnimalTriplet_Abstract):
    def __init__(self, config):
        super().__init__(config, mode="validation")
        self.augmentation = False


class AllAnimalTriplet_Train(AnimalTriplet_Abstract):
    def __init__(self, config):
        for animal in config["animals"]:
            dataroot = "synthetic_animals_triplet"
            try:
                self.data += AnimalTriplet_Train(dict(config, **{'dataroot': f'{dataroot}/{animal}s_meta'}))
            except:
                self.data = AnimalTriplet_Train(dict(config, **{'dataroot': f'{dataroot}/{animal}s_meta'}))

    def get_example(self, idx):
        return self.data.get_example(idx)


class AllAnimalTriplet_Validation(AnimalTriplet_Abstract):
    def __init__(self, config):
        for animal in config["animals"]:
            dataroot = "synthetic_animals_triplet"
            try:
                self.data += AnimalTriplet_Validation(dict(config, **{'dataroot': f'{dataroot}/{animal}s_meta'}))
            except:
                self.data = AnimalTriplet_Validation(dict(config, **{'dataroot': f'{dataroot}/{animal}s_meta'}))

    def get_example(self, idx):
        return self.data.get_example(idx)


class AllAnimalsVOC2011(AnimalTriplet_Abstract):
    def __init__(self, config):
        super().__init__(config, mode="all")
        for animal in config["animals"]:
            try:
                self.data += AnimalTriplet(dict(config, **{'dataroot': f'VOC2011/{animal}s_meta'}))
            except:
                self.data = AnimalTriplet(dict(config, **{'dataroot': f'VOC2011/{animal}s_meta'}))

    def get_example(self, idx):
        return self.data.get_example(idx)


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
