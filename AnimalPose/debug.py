import sys
import numpy as np
import torch
import torchvision
import PIL
import matplotlib.pyplot as plt
import cv2 as cv
import os

import flowiz as fz

import AnimalPose as ffg
from AnimalPose.data import get_dataloader_Human36MFramesFlow, Human36MFramesFlow
from AnimalPose.utils.flow_utils import (
    resample_bilinear,
    warp_flow_open_cv,
    warp_flow_open_cv_tensors,
    warp_image_pt,
    warp_image_pt_pwc,
)
import AnimalPose.models.resample2d_package.resample2d


class Options:
    def __init__(self, **entries):
        self.__dict__.update(entries)


opt = Options(
    flow_calc_mode="flownet2",
    dataroot="/net/hci-storage01/groupfolders/compvis/hperrot/datasets/human3.6M",
    flow_keep_intermediate=False,
    reverse_flow_input=True,
    flow_step=5,
    load_size=1024,
    max_dataset_size=sys.maxsize,
    isTrain=True,
    dataset_split_type="person",
)

loader = get_dataloader_Human36MFramesFlow(opt)

iterator = iter(loader)
# next(iterator)
ex2 = next(iterator)

model = ffg.models.SDCNet2D(opt)


def print_stats(return_dict, key):
    print(
        key,
        "min",
        return_dict[key].min().data.numpy(),
        "mean",
        return_dict[key].mean().data.numpy(),
        "max",
        return_dict[key].max().data.numpy(),
        "shape",
        return_dict[key].shape,
    )


def test_model():
    return_dict = model(ex2)

    print_stats(return_dict, "last_image")
    print_stats(return_dict, "target_image")
    print_stats(return_dict, "image_prediction")
    print_stats(return_dict, "flow_prediction")
    print_stats(return_dict, "input_flow")
    return return_dict


# _ = test_model()


def test_resample():
    return_dict = test_model()
    warped_dict = {}
    #     warped_dict['res_in_flow'] = resample_bilinear(return_dict['last_image'], return_dict['input_flow'])
    warped_dict["res_pred_flow"] = resample_bilinear(
        return_dict["last_image"], return_dict["flow_prediction"]
    )
    print()
    #     print_stats(warped_dict, 'res_in_flow')
    print_stats(warped_dict, "res_pred_flow")
    print()
    # resample2d = ffg.models.flownet2_pytorch.networks.resample2d_package.resample2d.Resample2d(bilinear=True)
    resample2d = AnimalPose.models.resample2d_package.resample2d.Resample2d(
        bilinear=True
    )
    warped_dict["res_in_flow_2d"] = resample2d(
        return_dict["last_image"], return_dict["input_flow"]
    )
    print_stats(warped_dict, "res_in_flow_2d")


# test_resample()


if __name__ == "__main__":
    import yaml

    config = {}
    # config_base_path = '/export/home/hperrot/pycharm_projects/AnimalPose/AnimalPose/configs'
    config_base_path = "/home/hperrot/src/AnimalPose/AnimalPose/configs"
    for file in ["config.yaml", "Human36MFramesFlow.yaml"]:
        config.update(yaml.read(os.path.join(config_base_path, file)))

    dataset = Human36MFramesFlow(config)
    ex = dataset[0]

    print(len(dataset))
