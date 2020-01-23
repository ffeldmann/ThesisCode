import torch

try:
    from AnimalPose.utils.loss_utils import update_loss_weights_inplace
except:
    import sys
    import os

    sys.path.append(os.path.dirname("."))
    from AnimalPose.utils.loss_utils import update_loss_weights_inplace

from AnimalPose.scripts.load_config import load_config
from AnimalPose.data import Human36MFramesFlowMeta_Train
from AnimalPose.utils.flow_utils import MaskCreatorForward


config = load_config("-b AnimalPose/configs/Human36MFramesFlowMeta.yaml".split(" "))
ds = Human36MFramesFlowMeta_Train(config)
ex = ds[0]


def test_update_loss_weights_inplace():
    loss_config = {
        "test_loss": {
            "start_ramp_it": 50000,
            "start_ramp_val": 0,
            "end_ramp_it": 60000,
            "end_ramp_val": 1,
        }
    }

    update_loss_weights_inplace(loss_config, 0)
    assert loss_config["test_loss"]["weight"] == 0

    update_loss_weights_inplace(loss_config, 50000)
    assert loss_config["test_loss"]["weight"] == 0

    update_loss_weights_inplace(loss_config, 55000)
    assert loss_config["test_loss"]["weight"] == 0.5

    update_loss_weights_inplace(loss_config, 57100)
    assert loss_config["test_loss"]["weight"] == 0.71

    update_loss_weights_inplace(loss_config, 60000)
    assert loss_config["test_loss"]["weight"] == 1

    update_loss_weights_inplace(loss_config, 100000)
    assert loss_config["test_loss"]["weight"] == 1


def test_MaskCreatorForward():
    # make sure it runs without errors
    mc = MaskCreatorForward(3, 0.5)

    flow_size = 256
    flow = torch.zeros(1, 2, flow_size, flow_size)
    flow[:, :, 10:15, 10:15] = 20

    mc(flow)


def test_MaskCreatorForward_dataset():
    mc = MaskCreatorForward(3, 0.5)

    flow = torch.tensor(ex["flow"]).unsqueeze(0).permute(0, 3, 1, 2)

    mask = mc(flow)


if torch.cuda.is_available():

    def test_MaskCreatorForward_dataset_cuda():
        mc = MaskCreatorForward(3, 0.5)
        flow = torch.tensor(ex["flow"]).unsqueeze(0).permute(0, 3, 1, 2)
        flow = flow.cuda()

        # mc.gaussian_filter.cuda()
        mask = mc(flow)

        assert mask.is_cuda
