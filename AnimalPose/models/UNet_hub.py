import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


class UNet_hub(nn.Module):
    def __init__(self, config):
        super(UNet_hub, self).__init__()
        self.n_channels = config["n_channels"]
        self.n_classes = config["n_classes"]

        self.model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
                    in_channels=self.n_channels, out_channels=self.n_classes, init_features=32, pretrained=config["pretrained"])

    def forward(self, x):
        return self.model(x)


class UNet_SMP(nn.Module):
    def __init__(self, config):
        super(UNet_SMP, self).__init__()
        self.n_channels = config["n_channels"]
        self.n_classes = config["n_classes"]
        aux_params = dict(
            pooling='avg',  # one of 'avg', 'max'
            dropout=0.5,  # dropout ratio, default is None
            activation='sigmoid',  # activation function, default is None
            classes=config["n_classes"],
        )
        self.model = smp.Unet('resnet34')
        self.model.segmentation_head[0] = nn.Conv2d(16, 20, kernel_size=3, stride=1, padding=1)


    def forward(self, x):
        return self.model(x)
