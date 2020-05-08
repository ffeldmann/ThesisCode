import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from AnimalPose.models import ResPoseNet


class Flatten(nn.Module):
    def forward(self, input):
        return input.reshape(input.size(0), -1)


class UnFlatten(nn.Module):
    def __init__(self, dim1, dim2):
        super(UnFlatten, self).__init__()
        self.dim1 = dim1
        self.dim1 = dim2

    def forward(self, input):
        return input.reshape(input.size(0), -1, self.dim1, self.dim2)


class AnimalPosenet(ResPoseNet):
    def __init__(self, config):
        super(AnimalPosenet, self).__init__(config)
        self.config = config
        self.variational = config["variational"]["active"]
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

        if config["encoder_2"]:
            self.backbone2 = self.backbone
        if config["resnet_type"] <= 38:
            # for resnet type 18, 38
            self.fc = nn.Linear(config["encoder_latent_dim"] * 2 if config["encoder_2"] else config["encoder_latent_dim"], 512 * 4 * 4)
        else:
            # For resnet type 50, 101, 152
            self.fc = nn.Linear(
                config["encoder_latent_dim"] * 2 if config["encoder_2"] else config["encoder_latent_dim"], 2048 * 4 * 4)
        if self.variational:
            self.fcmu = nn.Linear(config["encoder_latent_dim"], config["encoder_latent_dim"])
            self.fcvar = nn.Linear(config["encoder_latent_dim"], config["encoder_latent_dim"])

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x1, x2=None):
        if self.variational:
            x1 = self.backbone(x1)  # [1, 256]
            mu, logvar = self.fcmu(x1), self.fcvar(x1)
            x1 = self.reparameterize(mu, logvar)
            if x2 != None:
                x2 = self.backbone2(x2)
                x1x2 = torch.cat((x1, x2), dim=1)
                # Reshape x1 for Upsampling
                x1x2 = self.fc(x1x2).view(x1x2.size(0), -1, 4, 4)
                return self.head(x1x2), mu, logvar
            # Reshape x1 for Upsampling
            x1 = self.fc(x1).view(x1.size(0), -1, 4, 4)
            return self.head(x1), mu, logvar
        else:
            x1 = self.backbone(x1)
            if x2 != None:
                x2 = self.backbone2(x2)
                x1x2 = torch.cat((x1, x2), dim=1)
                # Reshape x1 for Upsampling
                x1x2 = self.fc(x1x2).view(x1.size(0), -1, 4, 4)
                return self.head(x1x2)
            # Reshape x1 for Upsampling
            x1 = self.fc(x1).view(x1.size(0), -1, 4, 4)
            return self.head(x1)


class AnimalEncoder(nn.Module):
    def __init__(self, config, variational=False):
        super(AnimalEncoder, self).__init__()
        self.config = config
        self.variational = variational

        model = getattr(models, "resnet" + str(config.get("resnet_type", "50")))(
            pretrained=config.get("pretrained", False))
        model.fc = nn.Linear(model.fc.in_features, config["encoder_latent_dim"])
        self.model = model
        # -> Output: [B, 256]

        if self.variational:
            self.fc1 = nn.Linear(config["encoder_latent_dim"], config["encoder_latent_dim"])
            self.fc2 = nn.Linear(config["encoder_latent_dim"], config["encoder_latent_dim"])

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + ((eps * std) * self.config["variational"]["kl_weight"])

    def forward(self, x):
        if self.variational:
            z = self.model(x)  # [B, 256]
            mu = z
            #        logvar = self.fc2(z)
            #        z = self.reparameterize(mu, logvar)
            z = z + torch.randn_like(z) * self.config["variational"]["kl_weight"]
            return z, mu, torch.ones_like(z)
        else:
            assert not self.variational
            return self.model(x)


class AnimalDecoder(nn.Module):
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
     P     |      --------+| |-------+ -      |  ~P
           |    -/         | |          \-    |
           |  -/           | |            \-  |
           |-/             +-+              \-|
                          concat
                            |                                   
           |-\              |
           |  -\            |
           |    -\          |
     A'    |      ----------+
           |    -/                                              
           |  -/                                                
           |-/


     All pre-trained models expect input images normalized in the same way, i.e. mini-batches of 3-channel RGB images of
     shape (3 x H x W), where H and W are expected to be at least 224. The images have to be loaded in to
     a range of [0, 1] and then normalized using mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225].
    """

    def __init__(self, config):
        super(AnimalNet, self).__init__()
        self.variational = config["variational"]["active"]
        self.encoder_x1 = AnimalEncoder(config, self.variational)
        if config["encoder_2"]:
            self.encoder_x2 = AnimalEncoder(config, False)
        self.decoder = AnimalDecoder(config)

    def forward(self, x1, x2=None):
        if self.variational:
            x1, mu, logvar = self.encoder_x1(x1)
            if x2 != None:
                x2 = self.encoder_x2(x2)
                mu2, logvar2 = None, None
                x1x2 = torch.cat((x1, x2), dim=1)
                # unsqueeze x1 adding [B, C, 1,1]
                x1x2 = x1x2.unsqueeze(-1).unsqueeze(-1)
                return self.decoder(x1x2), mu, logvar, mu2, logvar2
            x1 = x1.unsqueeze(-1).unsqueeze(-1)
            return self.decoder(x1), mu, logvar
        else:
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
