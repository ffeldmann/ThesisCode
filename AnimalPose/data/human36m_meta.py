import os
import numpy as np
import skimage
import flowiz as fz

from edflow.data.believers.meta import MetaDataset
from edflow.data.believers.meta_view import MetaViewDataset

from .util import get_frames_split, split_sequences, get_sequences, get_flow_folder


def flow_loader(path, factor, resize_to=None):
    """

    Parameters
    ----------
    path : str
        Where to find the flow.
    factor : float
        Factor, with which the flow is multiplied. This can adjust the value
        range of the tensors to match the load size.
    resize_to : list
        If not None, the loaded image will be resized to these dimensions. Must
        be a list of two integers or a single integer, which is interpreted as
        list of two integers with same value.

    Returns
    -------
    im : np.array
        A flow vector of an image loaded using :class:`PIL.Image` and adjusted to the range as
        specified.
    """

    def loader(factor=factor, resize_to=resize_to):
        flow = fz.read_flow(path) * factor

        if resize_to is not None:
            if isinstance(resize_to, int):
                resize_to = [resize_to] * 2
            if not list(resize_to) == list(flow.shape[:2]):
                flow = skimage.transform.resize(flow, resize_to)

        return flow

    return loader


class Human36MFramesMeta(MetaDataset):
    def __init__(self, config):
        self.config = config
        super().__init__(os.path.join(config["dataroot"], "meta", "images"))
        self.append_labels = True
        self.expand = True

    def __len__(self):
        return min(len(super()), self.config["max_dataset_size"])


class Human36MFramesFlowMeta(MetaViewDataset):
    def __init__(self, config, mode=None):
        self.config = config
        # only one meta dataset for forward and backward flow
        flow_folder = get_flow_folder(config, reverse_flow_input=False)
        super().__init__(os.path.join(config["dataroot"], "meta", flow_folder))
        self.append_labels = True
        self.base.append_labels = True
        self.expand = True

        self.split_type = config.get("dataset_split_type", "action")
        self.mode = mode or "train"
        if self.split_type == "none":
            self.mode = "all"

        assert self.split_type in ["person", "action", "none"]
        assert self.mode in ["train", "validation", "all"]

        # get data split
        split_indices = get_frames_split(self.base.labels, self.split_type, self.mode)
        sequences = self.views["images"]
        self.split_sequence_ids = np.where(np.isin(sequences[:, 0], split_indices))[0]

    def __len__(self):
        return min(len(self.split_sequence_ids), self.config["max_dataset_size"])


class Human36MFramesFlowMeta_Train(Human36MFramesFlowMeta):
    def __init__(self, config):
        super().__init__(config, mode="train")


class Human36MFramesFlowMeta_Validation(Human36MFramesFlowMeta):
    def __init__(self, config):
        super().__init__(config, mode="validation")


class Human36MFramesFlowMeta_ValidationFew(Human36MFramesFlowMeta_Validation):
    def __init__(self, config):
        self.config = config
        super().__init__(config)
        self.indices = np.array(
            [
                17104,
                348,
                9110,
                13392,
                9824,
                20801,
                2578,
                18749,
                18265,
                17801,
                17838,
                3964,
                3249,
                6831,
                13504,
                5000,
                9345,
                2837,
                6300,
                8084,
                10647,
                4851,
                20580,
                18876,
                11809,
                558,
                10218,
                10909,
                21092,
                6389,
                515,
                21129,
                3601,
                10430,
                18791,
                2021,
                11106,
                1024,
                9169,
                541,
                13885,
                3717,
                13202,
                1357,
                14461,
                4222,
                14606,
                4982,
                20261,
                35,
                6812,
                3060,
                20449,
                11500,
                4209,
                4177,
                21359,
                21312,
                7102,
                6435,
                10724,
                3284,
                18929,
                5864,
                20992,
                15010,
                20143,
                218,
                10135,
                6123,
                13963,
                17036,
                15075,
                9221,
                14208,
                16074,
                20012,
                9672,
                4672,
                20184,
                4398,
                20103,
                6610,
                9881,
                8293,
                2380,
                11890,
                14071,
                1014,
                19746,
                6463,
                11035,
                16335,
                3699,
                7277,
                14184,
                21087,
                6640,
                18347,
                15839,
            ]
        )  # some 100 quasi random indices

    def get_example(self, idx):
        idx = self.indices[idx]
        return super().get_example(idx)

    def __len__(self):
        return len(self.indices)


class Human36MFramesFlowMeta_FlowPrep(MetaViewDataset):
    def __init__(self, config):
        flow_folder = get_flow_folder(config, reverse_flow_input=False)
        super().__init__(os.path.join(config["dataroot"], "meta", flow_folder))

        self.append_labels = True
        self.base.append_labels = True
        self.base.expand = True
        self.base.loader_kwargs["image"]["resize_to"] = config["calc_size"]
        # don't load flow because it is not yet there
        del self.loaders["forward_flow"]
        del self.loaders["backward_flow"]
