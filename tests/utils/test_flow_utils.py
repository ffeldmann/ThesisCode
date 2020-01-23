import torch

from AnimalPose.utils.flow_utils import invert_flow_batch


def test_invert_flow_batch():
    flow = torch.zeros(2, 2, 6, 6)
    inverse = invert_flow_batch(flow)


if torch.cuda.is_available():

    def test_invert_flow_batch_cuda():
        flow = torch.zeros(2, 2, 6, 6).cuda()
        inverse = invert_flow_batch(flow)
        assert inverse.is_cuda
