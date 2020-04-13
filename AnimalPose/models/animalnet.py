import numpy as np
import torch.nn as nn
import torchvision.models as models
from torch.nn import PixelShuffle
import math
import torch
from edflow.custom_logging import get_logger

from collections import OrderedDict
from AnimalPose.models.utils import ICNR


class AnimalEncoder(nn.Module):
    def __init__(self, config, variational):
        super(AnimalEncoder, self).__init__()
        self.variational = variational

        self.logger = get_logger("Encoder")
        self.model = getattr(models, "resnet" + str(config.get("resnet_type", "50")))(
            pretrained=config.get("pretrained", False))
        self.config = config
        if config["load_encoder_pretrained"]["active"]:
            print(f"Loading weights for Encoder from {config['load_encoder_pretrained']['path']}.")
            state = torch.load(f"{config['load_encoder_pretrained']['path']}")
            try:
                self.model.fc = nn.Linear(state["model"]["encoder_x1.model.fc.weight"].shape[1],
                                          state["model"]["encoder_x1.model.fc.weight"].shape[0])
                new_state_dict = OrderedDict()
                for k, v in state["model"].items():
                    if k.startswith("encoder_"):
                        # remove `encoder_{x1,x2}.`
                        name = k.replace("encoder_x1.", "").replace("encoder_x2.", "").replace("model.", "")
                        new_state_dict[name] = v
                self.model.load_state_dict(new_state_dict)
                self.model.fc = nn.Linear(self.model.fc.in_features, config["encoder_latent_dim"])
            except Exception as exc:
                print(exc)
                new_state_dict = OrderedDict()
                for k, v in state["model"].items():
                    name = k.replace("model.", "")  # remove `model.`
                    new_state_dict[name] = v

                # Overrides default last layer with the shape of the pretrained
                # This layer is just adapted so we can load the weights without problems
                # It will be overwritten in the net step anyways.
                in_features = new_state_dict["fc.weight"].shape[1]
                classes = new_state_dict["fc.weight"].shape[0]
                self.model.fc = nn.Linear(in_features, classes)
                self.model.load_state_dict(new_state_dict)
        # save fc layer dimensions
        in_features = self.model.fc.in_features

        # resnet -> 256 -> mu, sigma
        if self.variational:
            self.model = nn.Sequential(*list(self.model.children())[:-1])
            self.fc1 = nn.Linear(in_features, config["encoder_latent_dim"])
            self.fc2 = nn.Linear(in_features, config["encoder_latent_dim"])
        else:
            self.model.fc = nn.Linear(in_features, config["encoder_latent_dim"])

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        enc = self.model(x)
        if self.variational:
            mu = self.fc1(enc.squeeze())
            logvar = self.fc2(enc.squeeze())
            enc = self.reparameterize(mu, logvar)
            return enc, mu, logvar
        return enc


class ResSequential(nn.Module):
    """
    Helper module for AnimalDecoderSubPixel
    """

    def __init__(self, layers, res_scale=1.0):
        super().__init__()
        self.res_scale = res_scale
        self.m = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.m(x) * self.res_scale


# we want Conv -> ReLu -> Conv
def conv(ni, nf, kernel_size=3, actn=False):
    """
    Helper for ResSequential
    Args:
        ni:
        nf:
        kernel_size:
        actn:

    Returns:

    """
    layers = nn.Sequential()
    layers.add_module("Conv2d", nn.Conv2d(ni, nf, kernel_size, padding=kernel_size // 2))
    if actn: layers.add_module("ReLu", nn.ReLU(inplace=True))
    return layers


def up_sample(ni, nf, scale, subpixel=False):
    """
    Helper for ResSequential
    Args:
        ni:
        nf:
        scale:

    Returns:

    """
    layers = nn.Sequential()
    if subpixel:
        for i in range(int(math.log(scale, 2))):
            layers.add_module("PixelConv", conv(ni, nf * 4))
            layers.add_module("PS", nn.PixelShuffle(2))
    else:
        layers.add_module("Up", nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(ni, nf, kernel_size=3, stride=1, padding=1),
        )
                          )
    return layers


def res_block(nf):
    """
    Helper for ResSequential
    Args:
        nf:

    Returns:

    """
    return ResSequential([conv(nf, nf, actn=True), conv(nf, nf)], 0.1)


class AnimalDecoder(nn.Module):
    # https://youtu.be/nG3tT31nPmQ?t=1965
    def __init__(self, config):
        super(AnimalDecoder, self).__init__()
        self.logger = get_logger(self)
        self.latent_size = config["encoder_latent_dim"] * 2 if config["encoder_2"] else config["encoder_latent_dim"]
        ipt_size = int(config["resize_to"])  # image size
        nc_out = config["n_channels"]  # output channels
        self.scale = 2
        self.superpixel = config["superpixel"]
        self.n_blocks = int(np.log2(ipt_size))  # no of blocks, last block is hand crafted
        complexity = int(self.latent_size)
        features = [conv(self.latent_size, complexity)]
        # BLOCKS 1 - N-1
        for i in range(1, self.n_blocks):
            c_out = int(complexity / 2 ** i)
            c_in = int(complexity / 2 ** (i - 1))
            for i2 in range(4):
                features.append(res_block(c_in))
            features.append(conv(c_in, c_out))
            features.append(nn.BatchNorm2d(c_out))
            features.append(up_sample(c_out, c_out, self.scale, self.superpixel))
        features += [conv(c_out, c_out),
                     nn.BatchNorm2d(c_out),
                     up_sample(c_out, c_out, self.scale),
                     conv(c_out, 3)]
        self.model = nn.Sequential(*features)

        def initialize_icnr(m):
            classname = m.__class__.__name__
            if classname == "Sequential":
                try:
                    m.PixelConv.Conv2d.weight.data.copy_(ICNR(m.PixelConv.Conv2d.weight.data))
                except:
                    pass

        if self.superpixel:
            self.model.apply(initialize_icnr)

    def forward(self, x):
        return self.model(x)


class AnimalDecoderOLD(nn.Module):
    def __init__(self, config):
        super(AnimalDecoder, self).__init__()

        self.latent_size = config["encoder_latent_dim"] * 2 if config["encoder_2"] else config["encoder_latent_dim"]
        ipt_size = int(config["resize_to"])  # image size
        complexity = 64
        nc_out = config["n_channels"]  # output channels
        norm = "bn"
        use_bias = True

        if norm is not None:
            if norm == "bn":
                norm = nn.BatchNorm2d
                use_bias = False
            elif norm == "in":  # Instance Norm
                norm = nn.InstanceNorm2d
                use_bias = True
            else:
                norm = None

        from torch.nn import ConvTranspose2d as CT

        self.n_blocks = int(np.log2(ipt_size) - 1)
        self.model = nn.Sequential()
        # BLOCK 0
        # bs x lat x 1 x 1 --> bs x cout x 4 x 4
        c_out = complexity * 2 ** (self.n_blocks - 2)
        self.model.add_module("b00", CT(self.latent_size, c_out, 4, 1, 0, bias=use_bias))
        if norm is not None and norm != "adain":
            self.model.add_module("b01", norm(c_out))
        self.model.add_module("b02", nn.LeakyReLU(0.15, inplace=True))

        kernel_size = 4
        stride = 2

        # BLOCKS 1 - N-1
        for i, b in enumerate(reversed(range(1, self.n_blocks - 1))):
            c_in = complexity * 2 ** (b)
            c_out = complexity * 2 ** (b - 1)
            n = "b" + str(b)
            self.model.add_module(
                n + "0",
                CT(c_in, c_out, kernel_size=kernel_size, stride=stride, padding=1, bias=use_bias),
            )
            if norm is not None and norm != "adain":
                self.model.add_module(n + "1", norm(c_out))
            self.model.add_module(n + "2", nn.LeakyReLU(0.15, inplace=True))

        # BLOCK N: 4 --> 1
        n = "b" + str(self.n_blocks - 1)
        s0, s1 = "0", "1"
        self.model.add_module(
            n + s0,
            CT(
                complexity, nc_out, kernel_size=kernel_size, stride=stride, padding=1, bias=use_bias
            ),
        )
        self.model.add_module(n + s1, nn.Tanh())

        if config["load_decoder_pretrained"]["active"]:
            model_dict = self.model.state_dict()
            state = torch.load(f"{config['load_decoder_pretrained']['path']}")
            new_state_dict = OrderedDict()
            for k, v in state["model"].items():
                if k.startswith("decoder"):
                    name = k.replace("decoder.", "").replace("model.", "")  # remove `decoder.`
                    new_state_dict[name] = v
            new_state_dict["b00.weight"] = model_dict["b00.weight"]
            self.model.load_state_dict(new_state_dict)

    def forward(self, x):
        return self.model(x)


class AnimalNet(nn.Module):
    """
	 +--------------------|L2|-------------------+              
     |                                           |              
     |                                           |              
     |     |-\             +-+              /-|  |              
     |     |  -\           | |            /-  |  |              
     |     |    -\         | |          /-    |  |              
     X     |      --------+| |-------+ -      |  ~x             
           |    -/         | |          \-    |
           |  -/           | |            \-  |
           |-/             +-+              \-|
                          concat
                            |                                   
           |-\              |
           |  -\            |
           |    -\          |
     x'    |      ----------+                                   
           |    -/                                              
           |  -/                                                
           |-/


     All pre-trained models expect input images normalized in the same way, i.e. mini-batches of 3-channel RGB images of
     shape (3 x H x W), where H and W are expected to be at least 224. The images have to be loaded in to
     a range of [0, 1] and then normalized using mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225].
    """

    def __init__(self, config):
        super(AnimalNet, self).__init__()
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        # self.normalize = torchvision.transforms.Normalize(mean=self.mean, std=self.std)
        self.config = config
        self.variational = True if config["variational"]["active"] else False
        self.encoder_x1 = AnimalEncoder(config, variational=self.variational)
        if config["encoder_2"]:
            self.encoder_x2 = AnimalEncoder(config, variational=self.variational)
        self.decoder = AnimalDecoder(config)

    def forward(self, x1, x2=None):
        # x1 = self.normalize(x1)
        if self.variational:
            x1, mu, logvar = self.encoder_x1(x1)
        else:
            x1 = self.encoder_x1(x1)
        if x2 != None:
            # x2 = self.normalize(x2)
            if self.variational:
                x2, mu2, logvar2 = self.encoder_x2(x2)
            else:
                x2 = self.encoder_x2(x2)
            x1x2 = torch.cat((x1, x2), dim=1)
            # unsqueeze x1 adding [B, C, 1,1]
            x1x2 = x1x2.unsqueeze(-1).unsqueeze(-1)
            if self.variational:
                return self.decoder(x1x2), mu, logvar, mu2, logvar2
            else:
                return self.decoder(x1x2)
        # unsqueeze x1 adding [B, C, 1,1]
        x1 = x1.unsqueeze(-1).unsqueeze(-1)
        if self.variational:
            return self.decoder(x1), mu, logvar
        else:
            return self.decoder(x1)
