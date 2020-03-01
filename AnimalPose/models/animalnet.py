import torch
import torch.nn as nn

from AnimalPose.models.resnet_vision import ResnetTorchVision


class AnimalEncoder(nn.Module):
    def __init__(self, config):
        super(AnimalEncoder, self).__init__()
        backbone = ResnetTorchVision(config)

        self.model = backbone

        self.model.add_module("Up", nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, config["n_classes"], kernel_size=3, stride=1,
                      padding=1),
        ),
                              )
        self.latent_vector = (nn.Sequential(
            nn.Linear(int(config["n_classes"]) * 128*128, 256),
            nn.ReLU(),
        ))

    def forward(self, x):
        x = self.model(x)
        x = x.view(x.size(0), -1) # reshape to [B, W*H*C]
        x = self.latent_vector(x)
        return x


class AnimalDecoder(nn.Module):
    def __init__(self, config):
        super(AnimalDecoder, self).__init__()
        self.model = None

    def forward(self, x):
        return self.model(x)


class AnimalNet(nn.Module):
    def __init__(self, config):
        super(AnimalNet, self).__init__()
        self.encoder_x1 = AnimalEncoder(config)
        #self.encoder_x2 = AnimalEncoder(config)
        self.decoder = AnimalDecoder(config)

    def forward(self, x1):
        x1 = self.encoder_x1(x1)
        #x2 = self.encoder_x2(x2)
        #x1x2 = torch.cat(x1, x2)
        return self.decoder(x1)
