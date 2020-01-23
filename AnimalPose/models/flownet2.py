import torch
import torch.nn as nn

import AnimalPose.models.flownet2_pytorch.models


class Options:
    def __init__(self, **entries):
        self.__dict__.update(entries)


class FlowNet2(AnimalPose.models.flownet2_pytorch.models.FlowNet2):
    def __init__(self, config):
        config.setdefault("rgb_max", 255)
        config.setdefault("fp16", False)
        args = Options(**config)
        super().__init__(args)

    def forward(self, inputs):
        inputs_fn = torch.stack([inputs["image"], inputs["target"]], dim=2)
        return super().forward(inputs_fn)


if __name__ == "__main__":
    config = {
        "rgb_max": 255.0,
        "fp16": False,
    }

    flownet2 = FlowNet2(config)

    print(flownet2)
