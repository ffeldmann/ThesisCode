import torch
import copy

from AnimalPose.utils.loss_utils import MaskedL1LossInstances
from AnimalPose.data.human36m_meta import Human36MFramesFlowMeta
from AnimalPose.scripts.load_config import load_config


config = load_config("-b AnimalPose/configs/Human36MFramesFlowMeta.yaml".split(" "))
config.update(dict(losses=dict(masked_L1=dict(mask_sigma=3, mask_threshold=0.5))))

ds = Human36MFramesFlowMeta(config)

ex = ds[0]
ex["np"] = {
    "image": ex["images"][0]["image"],
    "target": ex["images"][1]["image"],
    "flow": ex["flow"],
    "backward_flow": ex["backward_flow"],
}


def np2pt(array, cuda=False):
    tensor = torch.tensor(array).unsqueeze(0).permute(0, 3, 1, 2)
    if cuda:
        tensor = tensor.cuda()
    return tensor


def test_MaskedL1LossInstances():
    # make sure it runs without problems
    inputs = copy.deepcopy(ex)

    inputs["pt"] = {key: np2pt(inputs["np"][key], cuda=False) for key in inputs["np"]}

    Loss = MaskedL1LossInstances(config["losses"]["masked_L1"])

    loss = Loss(
        inputs["pt"]["image"],
        inputs["pt"]["target"],
        inputs["pt"]["flow"],
        inputs["pt"]["backward_flow"],
    )


if torch.cuda.is_available():

    def test_MaskedL1LossInstances_cuda():
        # make sure it runs without problems
        inputs = copy.deepcopy(ex)

        inputs["pt"] = {
            key: np2pt(inputs["np"][key], cuda=True) for key in inputs["np"]
        }

        Loss = MaskedL1LossInstances(config["losses"]["masked_L1"])

        loss = Loss(inputs["pt"]["image"], inputs["pt"]["target"], inputs["pt"]["flow"])
        assert loss.is_cuda
