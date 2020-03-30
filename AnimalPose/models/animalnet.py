import numpy as np
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms
import torch

from collections import OrderedDict
class AnimalEncoder(nn.Module):
    def __init__(self, config):
        super(AnimalEncoder, self).__init__()
        model = getattr(models, "resnet" + str(config.get("resnet_type", "50")))(
            pretrained=config.get("pretrained", False))
        if config["load_animal_pretrained"]["active"]:
            state = torch.load(f"{config['load_animal_pretrained']['encoder_path']}")

            try:
                model.fc = nn.Linear(model.fc.in_features, config["encoder_latent_dim"])
                new_state_dict = OrderedDict()
                for k, v in state["model"].items():
                    if k.startswith("encoder_"):
                        name = k[11:].replace("model.", "")  # remove `encoder_{x1,x2}.`
                        new_state_dict[name] = v
                model.load_state_dict(new_state_dict)
            except:

                new_state_dict = OrderedDict()
                for k, v in state["model"].items():
                    name = k[6:]  # remove `model.`
                    new_state_dict[name] = v

                # Overrides default last layer with the shape of the pretrained
                # This layer is just adapted so we can load the weights without problems
                # It will be overwritten in the net step anyways.
                in_features = new_state_dict["fc.weight"].shape[1]
                classes = new_state_dict["fc.weight"].shape[0]
                model.fc = nn.Linear(in_features, classes)
                model.load_state_dict(new_state_dict)
        model.fc = nn.Linear(model.fc.in_features, config["encoder_latent_dim"])
        self.model = model

    def forward(self, x):
        return self.model(x)


class AnimalDecoder(nn.Module):
    def __init__(self, config):
        super(AnimalDecoder, self).__init__()

        self.latent_size = config["decoder_latent_dim"] * 2 if config["encoder_2"] else config["decoder_latent_dim"]
        ipt_size = int(config["resize_to"].split(",")[0])  # image size
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

        if config["load_animal_pretrained"]["active"]:
            state = torch.load(f"{config['load_animal_pretrained']['decoder_path']}")
            new_state_dict = OrderedDict()
            for k, v in state["model"].items():
                if k.startswith("decoder"):
                    name = k[8:].replace("model.", "")  # remove `decoder.`
                    new_state_dict[name] = v
            self.model.load_state_dict(new_state_dict)
            #self.model.load_state_dict(state["model"])

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
    #normalize = torchvision.transforms.Normalize(mean=self.mean, std=self.std)

    def __init__(self, config):
        super(AnimalNet, self).__init__()
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.encoder_x1 = AnimalEncoder(config)
        if config["encoder_2"]:
            self.encoder_x2 = AnimalEncoder(config)
        self.decoder = AnimalDecoder(config)

    def forward(self, x1, x2=None):
        x1 = self.encoder_x1(x1)
        if x2 != None:
            x2 = self.encoder_x2(x2)
            x1x2 = torch.cat((x1, x2), dim=1)
            # unsqueeze x1 adding [B, C, 1,1]
            x1x2 = x1x2.unsqueeze(-1).unsqueeze(-1)
            return self.decoder(x1x2)
        # unsqueeze x1 adding [B, C, 1,1]
        x1 = x1.unsqueeze(-1).unsqueeze(-1)
        return self.decoder(x1)

# class ConvBlock(nn.Module):
#     """
#     Helper module that consists of a Conv -> BN -> ReLU
#     """
#
#     def __init__(self, in_channels, out_channels, padding=1, kernel_size=3, stride=1, with_nonlinearity=True):
#         super().__init__()
#         self.conv = nn.Conv2d(in_channels, out_channels, padding=padding, kernel_size=kernel_size, stride=stride)
#         self.bn = nn.BatchNorm2d(out_channels)
#         self.relu = nn.ReLU()
#         self.with_nonlinearity = with_nonlinearity
#
#     def forward(self, x):
#         x = self.conv(x)
#         x = self.bn(x)
#         if self.with_nonlinearity:
#             x = self.relu(x)
#         return x
#
#
# class ResidualBlock(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super().__init__()
#         self.in_channels, self.out_channels = in_channels, out_channels
#         self.blocks = nn.Identity()
#         self.activate = nn.ReLU()
#         self.shortcut = nn.Identity()
#
#     def forward(self, x):
#         residual = x
#         if self.should_apply_shortcut: residual = self.shortcut(x)
#         x = self.blocks(x)
#         x += residual
#         x = self.activate(x)
#         return x
#
#     @property
#     def should_apply_shortcut(self):
#         return self.in_channels != self.out_channels
