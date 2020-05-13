from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn
import torch
import torchvision.models as models
from edflow import get_logger
from torchvision.models.resnet import BasicBlock, Bottleneck
from torchvision.models.resnet import model_urls
from torch.utils import model_zoo


class ResnetTorchVisionKeypoints(nn.Module):
    def __init__(self, config):
        super(ResnetTorchVisionKeypoints, self).__init__()
        self.model = getattr(models, "resnet" + str(config.get("resnet_type", "50")))(
            pretrained=config.get("pretrained", False))
        self.input = nn.Sequential(*list(self.model.children())[:4])
        self.layer1 = self.model.layer1
        self.layer2 = self.model.layer2
        # Removing Last layer FC and AvgPool, as well as layer3 and layer4
        # self.model = nn.Sequential(*list(self.model.children())[:-4])

        if config["resnet_type"] == 18:
            self.layer3 = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),

                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            )
            self.layer4 = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.ReLU(inplace=True),
            )
            self.out = nn.Sequential(nn.Conv2d(32, config["n_classes"], kernel_size=3, stride=1, padding=1), )
        else:
            self.layer3 = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),

                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            )
            self.layer4 = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.ReLU(inplace=True),
            )
            self.out = nn.Sequential(nn.Conv2d(128, config["n_classes"], kernel_size=3, stride=1, padding=1), )

    def forward(self, x):
        x = self.input(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return self.out(x)


###################### Pose ResNet ###################

# Specification
resnet_spec = {
    18: (BasicBlock, [2, 2, 2, 2], [64, 64, 128, 256, 512], 'resnet18'),
    34: (BasicBlock, [3, 4, 6, 3], [64, 64, 128, 256, 512], 'resnet34'),
    50: (Bottleneck, [3, 4, 6, 3], [64, 256, 512, 1024, 2048], 'resnet50'),
    101: (Bottleneck, [3, 4, 23, 3], [64, 256, 512, 1024, 2048], 'resnet101'),
    152: (Bottleneck, [3, 8, 36, 3], [64, 256, 512, 1024, 2048], 'resnet152')
}


class ResNetBackbone(nn.Module):

    def __init__(self, block, layers, in_channel=3):
        self.inplanes = 64
        super(ResNetBackbone, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.normal_(m.weight, mean=0, std=0.001)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x


class DeconvHead(nn.Module):
    def __init__(self, in_channels, num_layers, num_filters, kernel_size, conv_kernel_size, num_joints, depth_dim,
                 with_bias_end=True):
        super(DeconvHead, self).__init__()

        conv_num_filters = num_joints * depth_dim

        assert kernel_size == 2 or kernel_size == 3 or kernel_size == 4, 'Only support kernel 2, 3 and 4'
        padding = 1
        output_padding = 0
        if kernel_size == 3:
            output_padding = 1
        elif kernel_size == 2:
            padding = 0

        assert conv_kernel_size == 1 or conv_kernel_size == 3, 'Only support kernel 1 and 3'
        if conv_kernel_size == 1:
            pad = 0
        elif conv_kernel_size == 3:
            pad = 1

        self.features = nn.ModuleList()
        for i in range(num_layers):
            _in_channels = in_channels if i == 0 else num_filters
            # self.features.append(
            #    nn.ConvTranspose2d(_in_channels, num_filters, kernel_size=kernel_size, stride=2, padding=padding,
            #                       output_padding=output_padding, bias=False))
            self.features.append(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))
            self.features.append(nn.Conv2d(_in_channels, num_filters,
                                           kernel_size=3, stride=1, padding=padding, bias=True))
            self.features.append(nn.BatchNorm2d(num_filters))
            self.features.append(nn.ReLU(inplace=True))

        if with_bias_end:
            self.features.append(
                nn.Conv2d(num_filters, conv_num_filters, kernel_size=conv_kernel_size, padding=pad, bias=True))
        else:
            self.features.append(
                nn.Conv2d(num_filters, conv_num_filters, kernel_size=conv_kernel_size, padding=pad, bias=False))
            self.features.append(nn.BatchNorm2d(conv_num_filters))
            self.features.append(nn.ReLU(inplace=True))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.001)
                if with_bias_end:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, mean=0, std=0.001)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        for idx, layer in enumerate(self.features):
            x = layer(x)
        return self.sigmoid(x)


class ResPoseNet(nn.Module):
    def __init__(self, config):
        super(ResPoseNet, self).__init__()

        num_deconv_layers = 5
        num_deconv_filters = 256
        num_deconv_kernel = 4
        final_conv_kernel = 1
        depth_dim = 1

        "resnet" + str(config.get("resnet_type", "50"))
        block_type, layers, channels, name = resnet_spec[int(config.get("resnet_type", "50"))]
        self.logger = get_logger(self)
        self.backbone = ResNetBackbone(block_type, layers)
        self.head = DeconvHead(
            channels[-1], num_deconv_layers,
            num_deconv_filters, num_deconv_kernel,
            final_conv_kernel, config['n_classes'], depth_dim
        )
        if config["pretrained"] and not config["load_self_pretrained_encoder"]["active"]:
            self.logger.info(f"Init {name} Network from model zoo.")
            org_resnet = model_zoo.load_url(model_urls[name])
            # drop orginal resnet fc layer, add 'None' in case of no fc layer, that will raise error
            org_resnet.pop('fc.weight', None)
            org_resnet.pop('fc.bias', None)
            self.backbone.load_state_dict(org_resnet)
        if config["load_self_pretrained_encoder"]["active"]:
            path = config["load_self_pretrained_encoder"]["path"]
            self.logger.info(f"Init encoder from pretraind path: {path}.")
            try:
                state_dict = torch.load(path, map_location="cuda")["model"]
                new_state_dict = {}
                for k, v in state_dict.items():
                    if k.startswith("backbone."):
                        name = k.replace("backbone.", "")
                        new_state_dict[name] = v
                new_state_dict.pop("layer4.fc.1.weight", None)
                new_state_dict.pop("layer4.fc.1.bias", None)
            except KeyError:
                state_dict = torch.load(path, map_location="cuda")
                new_state_dict = {}
                for k, v in state_dict.items():
                    if k.startswith("content_encoder."):
                        name = k.replace("content_encoder.", "")
                        new_state_dict[name] = v
                new_state_dict.pop("layer4.fc.1.weight", None)
                new_state_dict.pop("layer4.fc.1.bias", None)
                new_state_dict.pop("fc1.weight", None)
                new_state_dict.pop("fc1.bias", None)


            self.backbone.load_state_dict(new_state_dict, strict=True)

        if config["load_self_pretrained_decoder"]["active"]:
            path = config["load_self_pretrained_decoder"]["path"]
            self.logger.info(f"Init decoder from pretraind path: {path}.")
            state_dict = torch.load(path, map_location="cuda")["model"]
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith("head."):
                    name = k.replace("head.", "")
                    new_state_dict[name] = v
            new_state_dict["features.20.weight"] = self.head.features[20].weight
            new_state_dict["features.20.bias"] = self.head.features[20].bias
            self.head.load_state_dict(new_state_dict, strict=True)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        return self.sigmoid(x)
