import torch.nn as nn


class FlowFrameGenModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.input_channels = 3 + 2

    def warp(self, model_output, inputs):
        predictions = {}
        if self.warp_bilinear is None:
            predictions["image"] = model_output
        else:
            predictions["flow"] = model_output
            predictions["image"] = self.warp_bilinear(
                inputs["image"].contiguous(), model_output.contiguous()
            )
        return predictions
