import numpy as np
import skimage.color
from edflow.data.agnostics.subdataset import SubDataset
from edflow.data.believers.meta import MetaDataset
from edflow.data.dataset_mixin import DatasetMixin

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
            # To be filled
        ]


class AnimalVOC2011_Abstract(DatasetMixin):
    def __init__(self, config, mode="all"):
        assert mode in ["train", "validation", "all"], f"Should be train, validatiopn or all, got {mode}"
        self.sc = AnimalVOC2011(config)
        self.train = int(0.8 * len(self.sc))
        self.test = 1 - self.train
        self.sigma = config["sigma"]
        if mode != "all":
            split_indices = np.arange(self.train) if mode == "train" else np.arange(self.train + 1, len(self.sc))
            self.data = SubDataset(self.sc, split_indices)
        else:
            self.data = self.sc

    def get_example(self, idx):
        example = super().get_example(idx)
        image, keypoints = example["frames"](), self.labels["kps"][idx]
        if "crop" in self.data.data.config.keys():
            if self.data.data.config["crop"]:
                image, keypoints = crop(image, keypoints, self.labels["bboxes"][idx])
        # (H, W, C)
        image, keypoints = self.data.data.rescale(image, keypoints)

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
        example.pop("frames")  # TODO
        return example


class AnimalVOC2011_Train(AnimalVOC2011_Abstract):
    def __init__(self, config):
        super().__init__(config, mode="train")


class AnimalVOC2011_Validation(AnimalVOC2011_Abstract):
    def __init__(self, config):
        super().__init__(config, mode="validation")


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
