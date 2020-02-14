import numpy as np
import skimage.color
from edflow.data.agnostics.subdataset import SubDataset
from edflow.data.believers.meta import MetaDataset
from edflow.data.dataset_mixin import DatasetMixin

from AnimalPose.data.util import make_heatmaps, Rescale


class SingleCats(MetaDataset):
    def __init__(self, config):
        super().__init__(config["dataroot"])
        self.config = config
        #self.append_labels = False

        if "rescale_to" in self.config.keys():
            self.rescale = Rescale(self.config["rescale_to"])
        else:
            # Scaling to default size 128
            self.rescale = Rescale((128, 128))

        self.parts = {
            "Nose": 0,
            "LEye": 1,
            "LEar": 2,
            "REye": 3,
            "REar": 4,
            "LShoulder": 5,
            "LElbow": 6,
            "LWrist": 7,
            "RShoulder": 8,
            "RElbow": 9,
            "RWrist": 10,
            "LHip": 11,
            "LKnee": 12,
            "LAnkle": 13,
            "RHip": 14,
            "RKnee": 15,
            "RAnkle": 16,
            "TMiddle": 17,
            "TEnd": 18
        }
        self.joints = [
                [0, 1],  # Nose -> LEye
                [1, 2],  # LEye -> LEar
                [0, 3],  # Nose -> REye
                [3, 4],  # REye -> REar
                [3, 7],  # Nose -> LEye
                [0, 5],  # Nose -> LShoulder
                [0, 8],  # Nose -> RShoulder
                [5, 8],  # LShoulder -> RShoulder
                [5, 6],  # LShoulder -> LElbow
                [6, 7],  # LElbow -> LWrist
                [8, 9],  # RShoulder -> RElbow
                [9, 10],  # RElbow -> RWrist
                [5, 11],  # LShoulder -> LHip
                [8, 14],  # RShoulder -> RHip
                [11, 14],  # LHip -> RHip
                [11, 12],  # LHip -> LKnee
                [12, 13],  # LElbow -> LAnkle
                [14, 15],  # RHip -> RKnee
                [15, 16],  # RKnee -> RAnkle
                [14, 17],  # RHip -> TMiddle
                [11, 17],  # RHip -> TMiddle
                [17, 18],   # TMiddle -> TEnd
        ]


class SingleCatsUNet(DatasetMixin):
    def __init__(self, config, mode="all"):
        #super().__init__(config)
        assert mode in ["train", "validation", "all"], f"Should be train, validatiopn or all, got {mode}"
        self.sc = SingleCats(config)
        self.train = int(0.8 * len(self.sc))
        self.test = 1 - self.train

        if mode != "all":
            # TODO Better split e.g. split per video!
            split_indices = np.arange(self.train) if mode == "train" else np.arange(self.train+1, len(self.sc))
            self.data = SubDataset(self.sc, split_indices)
        else:
            self.data = self.sc

    def get_example(self, idx):
        example = super().get_example(idx)
        # (H, W, C)
        image, keypoints = self.data.data.rescale(example["frames"](), self.labels["kps"][idx])
        height = image.shape[0]
        width  = image.shape[1]
        if "as_grey" in self.data.data.config.keys():
            if self.data.data.config["as_grey"]:
                example["inp"] = skimage.color.rgb2gray(image).reshape(height, width, 1)
                assert(self.data.data.config["n_channels"] == 1), ("n_channels should be 1, got {}".format(self.data.data.config["n_channels"]))
            else:
                example["inp"] = image
        else:
            example["inp"] = image
        example["kps"] = keypoints
        example["targets"] = make_heatmaps(example["inp"], keypoints)
        example.pop("frames") # TODO
        return example

class SingleCatsUNet_Train(SingleCatsUNet):
    def __init__(self, config):
        super().__init__(config, mode="train")


class SingleCatsUNet_Validation(SingleCatsUNet):
    def __init__(self, config):
        super().__init__(config, mode="validation")


class SingleCatsDLC(SingleCats):
    def __init__(self, config):
        super().__init__(config)
    def get_example(self, idx):
        example = super().get_example(idx)
        return example

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

    DATAROOT = {"dataroot": '/export/home/ffeldman/Masterarbeit/data'}

    catsall = SingleCatsUNet(DATAROOT)
    #cats = SingleCatsUNet_Train(DATAROOT)
    cats = SingleCatsUNet_Validation(DATAROOT)
    ex = cats.get_example(3)
    for hm in ex["targets"]:
        print(hm.shape)
        plt.imshow(hm)
        plt.show()