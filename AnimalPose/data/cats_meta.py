import os
import numpy as np
import skimage.color
from edflow.data.believers.meta import MetaDataset
from AnimalPose.data.util import make_heatmaps, Rescale
#from edflow.data.believers.meta_view import MetaViewDataset


class SingleCats(MetaDataset):
    def __init__(self, config):
        super().__init__(config["dataroot"])
        self.config = config

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


class SingleCatsUNet(SingleCats):
    def __init__(self, config):
        super().__init__(config)

    def get_example(self, idx):
        example = super().get_example(idx)
        image, keypoints = self.rescale(example["frames"](), self.labels["kps"][idx])
        if "as_grey" in self.config.keys():
            example["inp"] = skimage.color.rgb2gray(image)
            assert(self.config["n_channels"] == 1), ("n_channels should be 1, got {}".format(self.config["n_channels"]))
        else:
            example["inp"] = image
        example["targets"] = make_heatmaps(example["inp"], keypoints)
        example.pop("frames")
        return example

class SingleCatsDLC(SingleCats):
    def __init__(self, config):
        super().__init__(config)
    def get_example(self, idx):
        example = super().get_example(idx)
        return example

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    cats = SingleCatsUNet({"dataroot": '/export/home/ffeldman/Masterarbeit/data'})
    ex = cats.get_example(3)
    for hm in ex["targets"]:
        print(hm.shape)
        plt.imshow(hm)
        plt.show()
