import torch
import torch.nn as nn
import torchvision

class Deeplabv3Resnet101_Hub(nn.Module):
    def __init__(self, config):
        super(Deeplabv3Resnet101_Hub, self).__init__()
        #self.model = torch.hub.load('pytorch/vision:v0.5.0', 'deeplabv3_resnet101', pretrained=config["pretrained"])
        self.model = torchvision.models.detection.keypointrcnn_resnet50_fpn(pretrained=True)
    def forward(self, x):
        return self.model(x)["out"]
