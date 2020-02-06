import os
import numpy as np
import skimage.color
from edflow.data.believers.meta import MetaDataset
from AnimalPose.data.util import make_heatmaps, Rescale
from edflow.data.agnostics.subdataset import SubDataset
from edflow.data.dataset_mixin import DatasetMixin

class AnimalVOC2011(MetaDataset):
    def __init__(self, config):
        super().__init__(config["dataroot"])
        self.config = config

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


