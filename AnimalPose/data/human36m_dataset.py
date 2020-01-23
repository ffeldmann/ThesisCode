import h5py
import os
import copy
import tqdm
import edflow
from edflow.data.dataset import DatasetMixin
from edflow.data.believers.sequence import SequenceDataset
from edflow.data.agnostics.subdataset import SubDataset
from PIL import Image
import numpy as np
from edflow import get_logger
from edflow.data.util import adjust_support
import AnimalPose


class Human36M(DatasetMixin):
    def __init__(self, config):
        self.logger = get_logger(self)
        self.config = config
        self.dataroot = config["dataroot"]
        h5_file = os.path.join(self.dataroot, "processed", "all", "annot.h5")
        self.labels = {}
        with h5py.File(h5_file, "r") as f:
            for k in f.keys():
                self.labels[k] = np.array(list(f[k][()]))

        def add_dataroot(filepath):
            return os.path.join(self.config["dataroot"], filepath.decode("utf-8"))

        self.labels["frame_path"] = np.array(
            list(map(add_dataroot, self.labels["frame_path"]))
        )
        self.logger.info("Added dataroot to frame paths")

        self.append_labels = True  # appends labels to returned example

        self.image_loader = edflow.data.believers.meta_loaders.image_loader

    def get_example(self, idx):
        path = self.labels["frame_path"][idx]
        return {
            "image": image_loader(
                path, support="-1->1", resize_to=[self.config["load_size"]] * 2
            )()
        }

    def __len__(self):
        return len(self.labels["frame_path"])


def get_flow_path(image_path):
    frame_number = image_path.split("/")[-1].split("_")[-1].split(".")[0]
    rel_save_path = os.path.join(
        image_path.split("/")[-5],  # person
        image_path.split("/")[-4],  # action
        image_path.split("/")[-2],  # camera
        "flow_" + frame_number + ".flo",  # frame
    )
    return rel_save_path


class Human36MFramesFlow(DatasetMixin):
    def __init__(self, config):
        super().__init__()
        self.config = copy.deepcopy(config)
        # init images
        ds_images = Human36M(config)
        ds_images.labels["index"] = np.arange(len(ds_images))
        # create sequence dataset
        ds_images_sequence = SequenceDataset(
            ds_images, length=2, step=config["target_frame_step"]
        )
        # remove account for base_frame_step
        frame_ids = ds_images_sequence.labels["fid"]
        frame_ids = np.where(frame_ids[:, 0] % config["base_frame_step"] == 0)[0]
        ds_image_sequence_base = SubDataset(ds_images_sequence, frame_ids)
        self.data = ds_image_sequence_base
        # set flow paths
        self.set_flow_paths()
        # loaders
        self.factor = self.config["load_size"] / self.config["calc_size"]
        self.resize_to = [self.config["load_size"]] * 2
        self.support = "-1->1"
        self.image_loader = edflow.data.believers.meta_loaders.image_loader
        self.flow_loader = AnimalPose.data.human36m_meta.flow_loader

    def set_flow_paths(self):
        for backward in [False, True]:
            flow_paths = list()
            for i in tqdm.tqdm(
                range(len(self)),
                desc="{} {}".format(self.config["target_frame_step"], backward),
            ):
                flow_paths.append(self.get_flow_path(i, backward))
            if backward:
                self.labels["backward_flow"] = np.array(flow_paths)
            else:
                self.labels["forward_flow"] = np.array(flow_paths)

    def get_flow_path(self, idx, backward):
        im_path = self.data.labels["frame_path"][idx][0]
        flow_folder = AnimalPose.data.util.get_flow_folder(self.config, backward)
        rel_flow_path = get_flow_path(im_path)
        flow_path = os.path.join(
            self.config["dataroot"], "flow", flow_folder, rel_flow_path
        )
        return flow_path

    def get_example(self, idx):
        # ex = super().get_example(idx)
        im_path = self.labels["frame_path"][idx]
        f_flow_path = self.labels["forward_flow"][idx]
        b_flow_path = self.labels["backward_flow"][idx]
        ex = {
            "images": [
                {
                    "image": self.image_loader(
                        im_path[0], support=self.support, resize_to=self.resize_to
                    )()
                },
                {
                    "image": self.image_loader(
                        im_path[1], support=self.support, resize_to=self.resize_to
                    )()
                },
            ],
            "forward_flow": self.flow_loader(
                f_flow_path, factor=self.factor, resize_to=self.resize_to
            )(),
            "backward_flow": self.flow_loader(
                b_flow_path, factor=self.factor, resize_to=self.resize_to
            )(),
        }
        return ex
