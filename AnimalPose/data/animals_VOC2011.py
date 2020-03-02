import imgaug.augmenters as iaa
import numpy as np
import skimage.color
from edflow.data.agnostics.subdataset import SubDataset
from edflow.data.believers.meta import MetaDataset
from edflow.data.dataset_mixin import DatasetMixin
from edflow.data.util import adjust_support

from AnimalPose.data.util import make_heatmaps, Rescale, crop


class AnimalVOC2011(MetaDataset):
    def __init__(self, config):
        super().__init__(config["dataroot"])
        self.config = config
        self.crop = crop
        if "rescale_to" in self.config.keys():
            self.rescale = Rescale(self.config["rescale_to"])
        else:
            # Scaling to default size 128
            self.rescale = Rescale((128, 128))


class AnimalVOC2011_Abstract(DatasetMixin):
    def __init__(self, config, mode="all"):
        assert mode in ["train", "validation", "all"], f"Should be train, validation or all, got {mode}"
        self.sc = AnimalVOC2011(config)
        self.train = int(config["train_size"] * len(self.sc))
        self.test = 1 - self.train
        self.sigma = config["sigma"]
        self.augmentation = config["augmentation"]
        if self.augmentation:
            self.seq = iaa.Sequential([
                #iaa.Sometimes(0.3, iaa.SaltAndPepper(0.01, per_channel=False)),
                #iaa.Sometimes(0.3, iaa.CoarseDropout(0.01, size_percent=0.5)),
                iaa.Sometimes(0.3, iaa.Fliplr(0.5)),
                iaa.Sometimes(0.3, iaa.Flipud(0.5)),
                #iaa.Sometimes(0.3,
                #            iaa.Affine(
                #                rotate=10,
                #                scale=(0.5, 0.7)
                #            )
                #),
            ])
        self.parts = {
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
            split_indices = np.arange(self.train) if mode == "train" else np.arange(self.train + 1, len(self.sc))
            self.data = SubDataset(self.sc, split_indices)
        else:
            self.data = self.sc

    def get_example(self, idx: object) -> object:
        """
        Args:
            idx: integer indicating index of dataset

        Returns: example element from dataset

        """
        example = super().get_example(idx)
        image, keypoints, bboxes = example["frames"]().astype(np.float32, copy=False), self.labels["kps"][idx], self.labels["bboxes"][idx]
        if "crop" in self.data.data.config.keys():
            if self.data.data.config["crop"]:
                image, keypoints = crop(image, keypoints, bboxes)
        if self.augmentation:
            # randomly perform some augmentations on the image, keypoints and bboxes
            image, keypoints = self.seq(image=image, keypoints=keypoints.reshape(1, -1, 2))
        # (H, W, C) and keypoints need to be reshaped from (N,J,2) -> (J,2)  J==Number of joints / keypoint pairs
        image, keypoints = self.data.data.rescale(image, keypoints.reshape(-1,2))
        height = image.shape[0]
        width = image.shape[1]
        if "as_grey" in self.data.data.config.keys():
            if self.data.data.config["as_grey"]:
                example["inp"] = skimage.color.rgb2gray(image).reshape(height, width, 1)
                assert (self.data.data.config["n_channels"] == 1), (
                    "n_channels should be 1, got {}".format(self.data.data.config["n_channels"]))
            else:
                example["inp"] = image
        else:
            example["inp"] = image
        example["kps"] = keypoints
        example["targets"] = make_heatmaps(example["inp"], keypoints, sigma=self.sigma)
        #example["joints"] = self.joints
        example.pop("frames")  # TODO: This removes the original frames which are not necessary for us here.
        return example


class AnimalVOC2011_Train(AnimalVOC2011_Abstract):
    def __init__(self, config):
        super().__init__(config, mode="train")


class AnimalVOC2011_Validation(AnimalVOC2011_Abstract):
    def __init__(self, config):
        super().__init__(config, mode="validation")
        self.augmentation = False


class AllAnimalsVOC2011_Train(AnimalVOC2011_Abstract):
    def __init__(self, config):
        #self.animals  = [AnimalVOC2011_Train(dict(config, **{'dataroot': f'VOC2011/{animal}s_meta'})) for animal in config["animals"]]
        for animal in config["animals"]:
            try:
                self.data += AnimalVOC2011_Train(dict(config, **{'dataroot': f'VOC2011/{animal}s_meta'}))
            except:
                self.data = AnimalVOC2011_Train(dict(config, **{'dataroot': f'VOC2011/{animal}s_meta'}))

    def get_example(self, idx):
        return self.data.get_example(idx)


class AllAnimalsVOC2011_Validation(AnimalVOC2011_Abstract):
    def __init__(self, config):
        for animal in config["animals"]:
            try:
                self.data += AnimalVOC2011_Validation(dict(config, **{'dataroot': f'VOC2011/{animal}s_meta'}))
            except:
                self.data = AnimalVOC2011_Validation(dict(config, **{'dataroot': f'VOC2011/{animal}s_meta'}))

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

    DATAROOT = {"dataroot": '/export/home/ffeldman/Masterarbeit/data/VOC2011/cats_meta'}
    cats = CatsVOC2011(DATAROOT)
    ex = cats.get_example(3)
    for hm in ex["targets"]:
        print(hm.shape)
        plt.imshow(hm)
        plt.show()
