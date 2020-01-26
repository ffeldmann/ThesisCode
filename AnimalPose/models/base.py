import torch.nn as nn


class FlowFrameGenModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.input_channels = 3 + 2

        # warping of flow
        warping_config = False #self.config["warping"]

        assert isinstance(warping_config, str)

        if "none" in warping_config:
            # predict RGB
            self.output_channels = 3
            self.warp_bilinear = None
        elif "backward_warp_gird_sample" in warping_config:
            from AnimalPose.utils.flow_utils import resample_bilinear

            self.output_channels = 2
            self.warp_bilinear = resample_bilinear
        elif "backward_warp_nvidia" in warping_config:
            from AnimalPose.models.flownet2_pytorch.networks.resample2d_package.resample2d import (
                Resample2d,
            )

            self.output_channels = 2
            self.warp_bilinear = Resample2d(bilinear=True)
        elif "forward_warp_plain" in warping_config:
            from AnimalPose.utils.flow_utils import forward_warp_permuted

            self.output_channels = 2
            self.warp_bilinear = forward_warp_permuted(interpolation_mode="Bilinear")
        elif "forward_warp_rescaling" in warping_config:
            from AnimalPose.utils.flow_utils import forward_warp_rescaled_permuted

            self.output_channels = 2
            self.warp_bilinear = forward_warp_rescaled_permuted()
        else:
            assert False, "Selection not valid {}. Chose one of {}".format(
                warping_config,
                [
                    "none",
                    "backward_warp_gird_sample",
                    "backward_warp_nvidia",
                    "forward_warp_plain",
                    "forward_warp_rescaling",
                ],
            )

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
