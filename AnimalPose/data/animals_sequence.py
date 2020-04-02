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

animal_class = {"cats": 0,
                "dogs": 1,
                "sheeps": 2,
                "cows": 3,
                "horses": 4,
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


class Animal_Sequence(MetaDataset):
    def __init__(self, config):
        super().__init__(config["dataroot"])
        self.config = config
        self.crop = crop


class Animal_Sequence_Abstract(DatasetMixin):
    def __init__(self, config, mode="all"):
        assert mode in ["train", "validation", "all"], f"Should be train, validation or all, got {mode}"
        self.config = config
        self.sc = SequenceDataset(Animal_Sequence(config), 2, step=config["sequence_step_size"])
        # works if dataroot like "VOC2011/cats_meta"
        # TODO PROBABLY NOT CORRECT HERE
        self.animal = config["dataroot"].split("/")[1].split("_")[0]

        self.train = int(config["train_size"] * len(self.sc))
        self.test = 1 - self.train
        self.sigma = config["sigma"]
        self.augmentation = config["augmentation"]
        self.aug_factor = 0.5

        if "rescale_to" in self.config.keys():
            self.rescale = Rescale(self.config["rescale_to"])
        else:
            # Scaling to default size 128
            self.rescale = Rescale((128, 128))
        if self.augmentation:
            self.seq = iaa.Sequential([
                iaa.Sometimes(self.aug_factor, iaa.AdditiveGaussianNoise(scale=0.05 * 255)),
                iaa.Sometimes(self.aug_factor, iaa.SaltAndPepper(0.01, per_channel=False)),
                iaa.Sometimes(self.aug_factor, iaa.CoarseDropout(0.01, size_percent=0.5)),
                iaa.Fliplr(self.aug_factor),
                iaa.Flipud(self.aug_factor),
                iaa.Sometimes(self.aug_factor,
                              iaa.Affine(
                                          rotate=10,
                                          scale=(0.5, 0.7)
                                        )
                              ),
                iaa.Sometimes(self.aug_factor, iaa.GaussianBlur(sigma=(0, 3.0))),
                iaa.LinearContrast((0.75, 1.5)),
                # Convert each image to grayscale and then overlay the
                # result with the original with random alpha. I.e. remove
                # colors with varying strengths.
                iaa.Grayscale(alpha=(0.0, 1.0)),
                iaa.Sometimes(self.aug_factor, iaa.Rain(speed=(0.1, 0.3))),
                iaa.Sometimes(self.aug_factor, iaa.Clouds()),
                iaa.Sometimes(self.aug_factor, iaa.MultiplyAndAddToBrightness(mul=(0.5, 1.5), add=(-30, 30))),
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
        output = dict()
        for ex_idx in range(self.data.data.length):
            image, keypoints, bboxes = example["frames"][ex_idx](), self.labels["kps"][idx][ex_idx], self.labels["bboxes"][idx][ex_idx]
            # store which keypoints are not present in the dataset
            zero_mask_x = np.where(keypoints[:, 0] <= 0)
            zero_mask_y = np.where(keypoints[:, 1] <= 0)
            # need uint 8 for augmentation methods
            image = adjust_support(image, "0->255")
            if "crop" in self.config.keys():
                if self.config["crop"]:
                    image, keypoints = crop(image, keypoints, bboxes)
            if self.augmentation:
                # randomly perform some augmentations on the image, keypoints and bboxes
                image, keypoints = self.seq(image=image, keypoints=keypoints.reshape(1, -1, 2))
            # (H, W, C) and keypoints need to be reshaped from (N,J,2) -> (J,2)  J==Number of joints / keypoint pairs
            image, keypoints = self.rescale(image, keypoints.reshape(-1, 2))
            keypoints[zero_mask_x] = np.array([0, 0])
            keypoints[zero_mask_y] = np.array([0, 0])
            # we always work with "0->1" images and np.float32
            # image = adjust_support(image, "0->1")
            height = image.shape[0]
            width = image.shape[1]
            if "as_grey" in self.config.keys():
                if self.config["as_grey"]:
                    output[f"inp{ex_idx}"] = adjust_support(skimage.color.rgb2gray(image).reshape(height, width, 1), "0->1")
                    assert (self.data.data.config["n_channels"] == 1), (
                        "n_channels should be 1, got {}".format(self.config["n_channels"]))
                else:
                    output[f"inp{ex_idx}"] = adjust_support(image, "0->1")
            else:
                output[f"inp{ex_idx}"] = adjust_support(image, "0->1")

            output[f"kps{ex_idx}"] = keypoints
            output[f"targets{ex_idx}"] = adjust_support(make_heatmaps(output[f"inp{ex_idx}"], keypoints, sigma=self.sigma), "0->1")
            output[f"animal_class"] = np.array(animal_class[self.animal])
            # example["joints"] = self.joints
            # example.pop("frames")  # TODO: This removes the original frames which are not necessary for us here.

        return output


class Animal_Sequence_Train(Animal_Sequence_Abstract):
    def __init__(self, config):
        super().__init__(config, mode="train")

class Animal_Sequence_Validation(Animal_Sequence_Abstract):
    def __init__(self, config):
        super().__init__(config, mode="validation")
        self.augmentation = False


class AllAnimals_Sequence_Train(Animal_Sequence_Abstract):
    def __init__(self, config):
        for animal in config["animals"]:
            try:
                self.data += Animal_Sequence_Train(dict(config, **{'dataroot': f'animals/{animal}s_meta'}))
            except:
                self.data = Animal_Sequence_Train(dict(config, **{'dataroot': f'animals/{animal}s_meta'}))

    def get_example(self, idx):
        return self.data.get_example(idx)


class AllAnimals_Sequence_Validation(Animal_Sequence_Abstract):
    def __init__(self, config):
        for animal in config["animals"]:
            try:
                self.data += Animal_Sequence_Validation(dict(config, **{'dataroot': f'animals/{animal}s_meta'}))
            except:
                self.data = Animal_Sequence_Validation(dict(config, **{'dataroot': f'animals/{animal}s_meta'}))

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

    # DATAROOT = {"dataroot": '/export/home/ffeldman/Masterarbeit/data/VOC2011/cats_meta'}
    # cats = CatsVOC2011(DATAROOT)
    # ex = cats.get_example(3)
    # for hm in ex["targets"]:
    #     print(hm.shape)
    #     plt.imshow(hm)
    #     plt.show()