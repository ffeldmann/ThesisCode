import torch.nn as nn
import torchvision.models as models


class ResnetTorchVisionClass(nn.Module):
    def __init__(self, config):
        super(ResnetTorchVisionClass, self).__init__()
        resnet = getattr(models, "resnet" + str(config.get("resnet_type", "50")))(
            pretrained=config.get("pretrained", False))
        num_ftrs = resnet.fc.in_features
        num_classes = 5
        resnet.fc = nn.Linear(num_ftrs, num_classes)
        self.model = resnet
    def forward(self, x):
        return self.model(x)
