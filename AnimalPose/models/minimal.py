"""
Portions of this code are adapted from:
https://github.com/NVIDIA/semantic-segmentation/blob/master/sdcnet/models/sdc_net2d.py
https://github.com/NVIDIA/flownet2-pytorch/blob/master/networks/FlowNetS.py
https://github.com/ClementPinard/FlowNetPytorch/blob/master/models/FlowNetS.py
"""
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from torch.nn import init
import time

from .model_utils import conv2d, deconv2d
from .base import FlowFrameGenModel
from edflow.util import retrieve
from edflow import get_logger


class Minimal(FlowFrameGenModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.logger = get_logger(self)

        self.logger.info("output channels: {}".format(self.output_channels))

        self.final_flow = nn.Conv2d(
            self.input_channels,
            self.output_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
        )

        # init parameters, when doing convtranspose3d, do bilinear init
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose3d):
                if m.bias is not None:
                    init.uniform_(m.bias)
                init.xavier_uniform_(m.weight)

        return

    def forward(self, inputs):

        before = time.time()

        # Network input
        images_and_flows = torch.cat((inputs["image"], inputs["flow"]), dim=1)

        output = self.final_flow(images_and_flows)

        if retrieve(self.config, "debug_timing", default=False):
            self.logger.info("calculations needed {} s".format(time.time() - before))

        return output
