import torch.nn as nn
import torchvision.models as models


class ResnetTorchVisionKeypoints(nn.Module):
    def __init__(self, config):
        super(ResnetTorchVisionKeypoints, self).__init__()
        resnet = getattr(models, "resnet" + str(config.get("resnet_type", "50")))(
            pretrained=config.get("pretrained", False))
        if "resnet_type" == "18":
            resnet.layer3 = nn.Sequential(
                                          nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
                                          nn.ReLU(inplace=True),
                                          nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
                                          nn.ReLU(inplace=True),
                                         )
            resnet.layer4 = nn.Sequential(
                                          nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
                                          )
            self.model = resnet
            # Replacing the last fully connected layer with a conv layer
            self.model = nn.Sequential(*list(resnet.children())[:-2])
            self.model.add_module("Out",
                                  nn.Sequential(
                                                nn.Conv2d(32, config["n_classes"], kernel_size=3, stride=1, padding=1),
                                                )
                                  )
        else:
            resnet.layer3 = nn.Sequential(
                nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.ReLU(inplace=True),
            )
            resnet.layer4 = nn.Sequential(
                nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.ReLU(inplace=True),
            )
            self.model = resnet
            # Replacing the last fully connected layer with a conv layer
            self.model = nn.Sequential(*list(resnet.children())[:-2])
            self.model.add_module("Out",
                                  nn.Sequential(
                                      nn.Conv2d(64, config["n_classes"], kernel_size=3, stride=1, padding=1),
                                      nn.ReLU(inplace=True),
                                      )
                                  )

    #o = (i -1)*s - 2*p + k + output_padding
    def forward(self, x):
        return self.model(x)
