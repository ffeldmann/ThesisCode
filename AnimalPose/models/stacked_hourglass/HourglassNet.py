import torch
import torch.nn as nn
from .hourglass_layers import Conv, Hourglass, Pool, Residual
from .loss import HeatmapLoss


class UnFlatten(nn.Module):
    def forward(self, input):
        return input.view(-1, 256, 4, 4)


class Merge(nn.Module):
    def __init__(self, x_dim, y_dim):
        super(Merge, self).__init__()
        self.conv = Conv(x_dim, y_dim, 1, relu=False, bn=False)

    def forward(self, x):
        return self.conv(x)


class HourglassNet(nn.Module):
    def __init__(self, config):
        super(HourglassNet, self).__init__()
        bn = False
        increase = 0
        self.nstack = config["nstack"]
        self.pre = nn.Sequential(
            Conv(3, 64, 7, 2, bn=True, relu=True),
            Residual(64, 128),
            Pool(2, 2),
            Residual(128, 128),
            Residual(128, config["inp_dim"])
        )

        self.hgs = nn.ModuleList([
            nn.Sequential(
                Hourglass(4, config["inp_dim"], bn, increase),
            ) for i in range(config["nstack"])])

        self.features = nn.ModuleList([
            nn.Sequential(
                Residual(config["inp_dim"], config["inp_dim"]),
                Conv(config["inp_dim"], config["inp_dim"], 1, bn=True, relu=True)
            ) for i in range(config["nstack"])])

        self.outs = nn.ModuleList([Conv(config["inp_dim"], config["n_classes"], 1, relu=False, bn=False) for i in range(config["nstack"])])
        self.merge_features = nn.ModuleList([Merge(config["inp_dim"], config["inp_dim"]) for i in range(config["nstack"] - 1)])
        self.merge_preds = nn.ModuleList([Merge(config["n_classes"], config["inp_dim"]) for i in range(config["nstack"] - 1)])
        self.heatmapLoss = HeatmapLoss()

    def forward(self, imgs):
        ## our HourglassNet
        #x = imgs.permute(0, 3, 1, 2)  # x of size 1,3, inpdim,inpdim
        x = self.pre(imgs) # preprocessing??
        combined_hm_preds = []
        for i in range(self.nstack):
            hg = self.hgs[i](x)
            feature = self.features[i](hg)
            preds = self.outs[i](feature)
            combined_hm_preds.append(preds)
            if i < self.nstack - 1:
                x = x + self.merge_preds[i](preds) + self.merge_features[i](feature)
        return torch.stack(combined_hm_preds, 1)

    def calc_loss(self, combined_hm_preds, heatmaps):
        combined_loss = []
        for i in range(self.nstack):
            combined_loss.append(self.heatmapLoss(combined_hm_preds[0][:, i], heatmaps))
        combined_loss = torch.stack(combined_loss, dim=1)
        return combined_loss
