import torch
import torch.nn as nn

class UNet_hub(nn.Module):
    def __init__(self, config):
        super(UNet_hub, self).__init__()
        self.n_channels = config["n_channels"]
        self.n_classes = config["n_classes"]

        self.model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
                    in_channels=self.n_channels, out_channels=self.n_classes, init_features=32, pretrained=False)

    def forward(self, x):
        return self.model(x)
