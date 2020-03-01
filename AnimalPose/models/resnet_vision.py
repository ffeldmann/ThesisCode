import torch.nn as nn
import torchvision.models as models


class ResnetTorchVision(nn.Module):
    def __init__(self, config):
        super(ResnetTorchVision, self).__init__()
        # resnet = resnet_fpn_backbone('resnet50', False)
        resnet = getattr(models, "resnet" + str(config.get("resnet_type", "50")))(
            pretrained=config.get("pretrained", False))
        resnet.conv1 = nn.Conv2d(config["n_channels"], 64, kernel_size=7, stride=2, padding=3,
                                 bias=False)
        nn.UpsamplingBilinear2d()
        resnet.layer3 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1)
                                      )
        resnet.layer4 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
                                      )
        self.model = resnet
        # Replacing the last fully connected layer with a conv layer
        self.model = nn.Sequential(*list(resnet.children())[:-2])
        self.model.add_module("Up", nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                                  nn.ReLU(inplace=True),
                                                  nn.Conv2d(128, config["n_classes"], kernel_size=3, stride=1,
                                                            padding=1)
                                                  )
                              )

    def forward(self, x):
        return self.model(x)
