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


class SDCNet2D(FlowFrameGenModel):
    def __init__(self, config):
        super().__init__(config)
        self.logger = get_logger(self)

        self.rgb_max = 1  # args.rgb_max

        factor = 2

        self.logger.info("output channels: {}".format(self.output_channels))

        self.conv1 = conv2d(self.input_channels, 64 // factor, kernel_size=7, stride=2)
        self.conv2 = conv2d(64 // factor, 128 // factor, kernel_size=5, stride=2)
        self.conv3 = conv2d(128 // factor, 256 // factor, kernel_size=5, stride=2)
        self.conv3_1 = conv2d(256 // factor, 256 // factor)
        self.conv4 = conv2d(256 // factor, 512 // factor, stride=2)
        self.conv4_1 = conv2d(512 // factor, 512 // factor)
        self.conv5 = conv2d(512 // factor, 512 // factor, stride=2)
        self.conv5_1 = conv2d(512 // factor, 512 // factor)
        self.conv6 = conv2d(512 // factor, 1024 // factor, stride=2)
        self.conv6_1 = conv2d(1024 // factor, 1024 // factor)

        self.deconv5 = deconv2d(1024 // factor, 512 // factor)
        self.deconv4 = deconv2d(1024 // factor, 256 // factor)
        self.deconv3 = deconv2d(768 // factor, 128 // factor)
        self.deconv2 = deconv2d(384 // factor, 64 // factor)
        self.deconv1 = deconv2d(192 // factor, 32 // factor)
        self.deconv0 = deconv2d(96 // factor, 16 // factor)

        self.final_flow = nn.Conv2d(
            self.input_channels + 16 // factor,
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

        self.ignore_keys = ["flownet2"]
        return

    def network_output(self, input_image, input_flow):

        # Network input
        images_and_flows = torch.cat((input_flow, input_image), dim=1)

        # encoder
        out_conv1 = self.conv1(images_and_flows)

        out_conv2 = self.conv2(out_conv1)
        out_conv3 = self.conv3_1(self.conv3(out_conv2))
        out_conv4 = self.conv4_1(self.conv4(out_conv3))
        out_conv5 = self.conv5_1(self.conv5(out_conv4))
        out_conv6 = self.conv6_1(self.conv6(out_conv5))

        # decoder with skip connections
        out_deconv5 = self.deconv5(out_conv6)
        concat5 = torch.cat((out_conv5, out_deconv5), 1)

        out_deconv4 = self.deconv4(concat5)
        concat4 = torch.cat((out_conv4, out_deconv4), 1)

        out_deconv3 = self.deconv3(concat4)
        concat3 = torch.cat((out_conv3, out_deconv3), 1)

        out_deconv2 = self.deconv2(concat3)
        concat2 = torch.cat((out_conv2, out_deconv2), 1)

        out_deconv1 = self.deconv1(concat2)
        concat1 = torch.cat((out_conv1, out_deconv1), 1)

        out_deconv0 = self.deconv0(concat1)

        # concatenate input before fianl convolution
        concat0 = torch.cat((images_and_flows, out_deconv0), 1)
        output_flow = self.final_flow(concat0)

        return output_flow

    def forward(self, inputs):

        before = time.time()

        output = self.network_output(inputs["image"], inputs["flow"])

        if retrieve(self.config, "debug_timing", default=False):
            self.logger.info("calculations needed {} s".format(time.time() - before))

        return output
