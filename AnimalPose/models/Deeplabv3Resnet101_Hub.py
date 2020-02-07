import torch
import torch.nn as nn

class Deeplabv3Resnet101_Hub(nn.Module):
    def __init__(self, config):
        super(Deeplabv3Resnet101_Hub, self).__init__()
        self.model = torch.hub.load('pytorch/vision:v0.5.0', 'deeplabv3_resnet101', pretrained=config["pretrained"])

    def forward(self, x):
        return self.model(x)
