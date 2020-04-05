import torch.nn as nn
import torchvision.models as models


class ResnetTorchVisionKeypoints(nn.Module):
    def __init__(self, config):
        super(ResnetTorchVisionKeypoints, self).__init__()
        resnet = getattr(models, "resnet" + str(config.get("resnet_type", "50")))(
            pretrained=config.get("pretrained", False))
        resnet.layer3 = nn.Sequential(
                                      nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=0),
                                      nn.ReLU(inplace=True),
                                      nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=0),
                                      nn.ReLU(inplace=True),
                                     )
        resnet.layer4 = nn.Sequential(
                                      nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=0),
                                      nn.ReLU(inplace=True),
                                      )
        self.model = resnet
        # Replacing the last fully connected layer with a conv layer
        self.model = nn.Sequential(*list(resnet.children())[:-2])
        self.model.add_module("Out",
                              nn.Sequential(
                                            nn.Conv2d(32, config["n_classes"], kernel_size=3, stride=1, padding=0),
                                            nn.Tanh(),
                                            )
                              )

    def forward(self, x):
        return self.model(x)
