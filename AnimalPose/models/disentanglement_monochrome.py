import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from AnimalPose.models import ResPoseNet


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class UnFlatten(nn.Module):
    def __init__(self, dim1, dim2):
        super(UnFlatten, self).__init__()
        self.dim1 = dim1
        self.dim1 = dim2

    def forward(self, input):
        return input.view(input.size(0), -1, self.dim1, self.dim2)


class DisentangleMonochrome(ResPoseNet):
    def __init__(self, config):
        super(DisentangleMonochrome, self).__init__(config)
        self.config = config
        if config["load_self_pretrained_encoder"]["active"]:
            path = config["load_self_pretrained_encoder"]["path"]
            self.logger.info(f"Load self pretrained encoder from {path}")
            state_dict = torch.load(path, map_location="cuda")["model"]
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith("backbone"):
                    name = k.replace("backbone.", "")
                    new_state_dict[name] = v
            self.backbone.load_state_dict(new_state_dict, strict=False)

        if config["load_self_pretrained_decoder"]["active"]:
            path = config["load_self_pretrained_decoder"]["path"]
            self.logger.info(f"Load self pretrained decoder from {path}")
            state_dict = torch.load(path, map_location="cuda")["model"]
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith("head"):
                    name = k.replace("head.", "")
                    new_state_dict[name] = v
            new_state_dict.pop("features.20.weight")
            new_state_dict.pop("features.20.bias")
            self.head.load_state_dict(new_state_dict, strict=False)
        self.head.features[20] = nn.Conv2d(256, 3, kernel_size=1, stride=1)

        # resnet 18 / 34 need different input resnet 50/101/152 : 2048
        if config["resnet_type"] <= 38:
            self.backbone.layer4.add_module("fc", nn.Sequential(
                Flatten(),
                nn.Linear(512 * 4 * 4, config["encoder_latent_dim"])
            ))
        else:
            self.backbone.layer4.add_module("fc", nn.Sequential(
                Flatten(),
                nn.Linear(2048 * 4 * 4, config["encoder_latent_dim"])
            )
                                            )

        self.backbone2 = self.backbone

        if config["resnet_type"] <= 38:
            # for resnet type 18, 38
            self.fc = nn.Linear(config["encoder_latent_dim"] * 2, 512 * 4 * 4)
        else:
            # For resnet type 50, 101, 152
            self.fc = nn.Linear(config["encoder_latent_dim"] * 2, 2048 * 4 * 4)

    def forward(self, input=None, input2=None, enc_appearance=None, enc_pose=None, mixed_reconstruction=False, cycle=False):
        # FIRST ALWAYS APPEARANCE; SECOND ALWAYS POSE!
        # "backbone" pose encoder
        # "backbone2" appearance encoder

        if enc_appearance != None and enc_pose != None and mixed_reconstruction:
            # mix reconstruction (b)
            latent = torch.cat((enc_appearance, enc_pose), dim=1)
            latent = self.fc(latent).view(latent.size(0), -1, 4, 4)
            return self.head(latent)

        appearance = self.backbone2(input)
        if input2 != None:
            pose = self.backbone(input)
        else:
            pose = self.backbone(input)

        if enc_appearance != None and enc_pose != None and cycle:
            # cycle consistency (c)

            latent_hand = torch.cat((enc_appearance, pose), dim=1)
            latent_hand = self.fc(latent_hand).view(latent_hand.size(0), -1, 4, 4)
            recon_hand = self.head(latent_hand)

            latent_thumb = torch.cat((appearance, enc_pose), dim=1)
            latent_thumb = self.fc(latent_thumb).view(latent_thumb.size(0), -1, 4, 4)
            recon_thumb = self.head(latent_thumb)

            return recon_hand, recon_thumb, appearance, pose

        latent = torch.cat((appearance, pose), dim=1)
        # Reshape latent for Upsampling
        latent = self.fc(latent).view(latent.size(0), -1, 4, 4)
        return self.head(latent), appearance, pose
