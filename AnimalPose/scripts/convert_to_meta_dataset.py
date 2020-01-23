import numpy as np
import tqdm
import copy
import yaml
import h5py
import shutil

import os
import sys

sys.path.append(os.path.dirname("."))

from edflow.data.dataset_mixin import DatasetMixin

import AnimalPose.data
from AnimalPose.data.util import get_flow_folder
from AnimalPose.scripts.load_config import load_config


config_ = load_config()
print(config_)


def store_meta(
    labels, meta_root, loaders=None, ignore_keys=None, rename_keys=None, meta=None
):
    loaders = loaders or dict()
    ignore_keys = ignore_keys or list()
    rename_keys = rename_keys or dict()

    from edflow.data.believers.meta_util import store_label_mmap

    for key, value in labels.items():
        if key in ignore_keys:
            continue
        if key in rename_keys:
            key = rename_keys[key]
        shape_str = "x".join([str(dim) for dim in value.shape])
        type_str = str(value.dtype)
        if key in loaders:
            key += ":" + loaders[key]
        file_str = "-*-".join([key, shape_str, type_str]) + ".npy"

        file_str = os.path.join(meta_root, "labels", file_str)

        os.makedirs(os.path.dirname(file_str), exist_ok=True)
        fp = np.memmap(file_str, dtype=type_str, mode="w+", shape=value.shape)

        fp[:] = value[:]

    if meta is not None:
        with open(os.path.join(meta_root, "meta.yaml"), mode="w+") as f:
            yaml.dump(meta, f)


def store_meta_images(config):
    dataset = AnimalPose.data.human36m_dataset.Human36M(config)
    loaders = {
        "frame_path": "image",
    }
    labels = copy.deepcopy(dataset.labels)

    meta = {
        "description": "This dataset loads the Human3.6M images.",
        "loader_kwargs": {
            "image": {"support": "-1->1", "resize_to": config["load_size"]}
        },
    }
    # store meta images
    store_meta(
        labels,
        meta_root=os.path.join(config["dataroot"], "meta/images"),
        loaders={"image": "image"},
        ignore_keys=["pid"],
        rename_keys={"frame_path": "image", "subject": "subject_id",},
        meta=meta,
    )


def store_meta_sequences(config):
    dataset = AnimalPose.data.human36m_dataset.Human36MFramesFlow(config)
    labels = {
        "sequence_view": dataset.labels["index"],
        "forward_flow": dataset.labels["forward_flow"],
        "backward_flow": dataset.labels["backward_flow"],
    }
    meta = {
        "description": f"This dataset loads images and flows from the the Human3.6M images with frame_lag={config['target_frame_step']}.",
        "base_dset": "edflow.data.believers.meta.MetaDataset",
        "base_kwargs": {"root": os.path.join(config["dataroot"], "meta", "images")},
        "loaders": {
            "forward_flow": "AnimalPose.data.human36m_meta.flow_loader",
            "backward_flow": "AnimalPose.data.human36m_meta.flow_loader",
        },
        "loader_kwargs": {
            "forward_flow": {
                "factor": config["load_size"] / config["calc_size"],
                "resize_to": config["load_size"],
            },
            "backward_flow": {
                "factor": config["load_size"] / config["calc_size"],
                "resize_to": config["load_size"],
            },
        },
        "views": {"images": "sequence_view"},
    }
    meta_root = os.path.join(
        config["dataroot"], "meta", get_flow_folder(config, reverse_flow_input=False)
    )
    store_meta(
        labels, meta_root=meta_root, meta=meta,
    )

    with open(os.path.join(meta_root, "meta.yaml"), mode="w+") as f:
        yaml.dump(meta, f)

    # copy dataset implementation
    for implementation_file in [
        AnimalPose.data.human36m_meta.__file__,
        AnimalPose.data.util.__file__,
    ]:
        src_file = implementation_file
        dst_file = os.path.join(meta_root, os.path.basename(src_file))
        shutil.copyfile(src_file, dst_file)


if __name__ == "__main__":

    config = copy.deepcopy(config_)

    if True:  # Create meta datasets
        if True:  # Create images meta dataset
            store_meta_images(config)
        if True:  # Create flow meta dataset
            for target_frame_step in [25]:
                config.update(target_frame_step=target_frame_step)
                store_meta_sequences(config)

    if True:
        test_command = "python -m pytest tests/data"
        print(test_command)
        os.system(test_command)
