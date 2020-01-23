import torch

if torch.cuda.is_available():

    def test_cuda_available():
        assert torch.cuda.is_available()
        a = torch.tensor(1)
        a.cuda()

    def test_forward_warp_cuda_available():
        import forward_warp_cuda

    def test_flownet2_cuda_available():
        import resample2d_cuda
        import correlation_cuda
        import channelnorm_cuda
