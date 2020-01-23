# Code adapted from https://raw.githubusercontent.com/dancelogue/flownet2-pytorch/master/utils/flow_utils.py

import numpy as np
import cv2 as cv
import torch
import torchfields
import kornia


TAG_CHAR = np.array([202021.25], np.float32)


def readFlow(fn):
    """ Read .flo file in Middlebury format"""
    # Code adapted from:
    # http://stackoverflow.com/questions/28013200/reading-middlebury-flow-files-with-python-bytes-array-numpy

    # WARNING: this will work on little-endian architectures (eg Intel x86) only!
    # print 'fn = %s'%(fn)
    with open(fn, "rb") as f:
        magic = np.fromfile(f, np.float32, count=1)
        if 202021.25 != magic:
            print("Magic number incorrect. Invalid .flo file")
            return None
        else:
            w = np.fromfile(f, np.int32, count=1)
            h = np.fromfile(f, np.int32, count=1)
            # print 'Reading %d x %d flo file\n' % (w, h)
            data = np.fromfile(f, np.float32, count=2 * int(w) * int(h))
            # Reshape data into 3D array (columns, rows, bands)
            # The reshape here is for visualization, the original code is (w,h,2)
            return np.resize(data, (int(h), int(w), 2))


def writeFlow(filename, uv, v=None):
    """ Write optical flow to file.
    
    If v is None, uv is assumed to contain both u and v channels,
    stacked in depth.
    Original code by Deqing Sun, adapted from Daniel Scharstein.
    """
    nBands = 2

    if v is None:
        assert uv.ndim == 3
        assert uv.shape[2] == 2
        u = uv[:, :, 0]
        v = uv[:, :, 1]
    else:
        u = uv

    assert u.shape == v.shape
    height, width = u.shape
    f = open(filename, "wb")
    # write the header
    f.write(TAG_CHAR)
    np.array(width).astype(np.int32).tofile(f)
    np.array(height).astype(np.int32).tofile(f)
    # arrange into matrix form
    tmp = np.zeros((height, width * nBands))
    tmp[:, np.arange(width) * 2] = u
    tmp[:, np.arange(width) * 2 + 1] = v
    tmp.astype(np.float32).tofile(f)
    f.close()


def warp_flow_open_cv(img, flow):
    h, w = flow.shape[:2]
    flow = flow
    flow[:, :, 0] += np.arange(w)
    flow[:, :, 1] += np.arange(h)[:, np.newaxis]
    res = cv.remap(img, flow, None, cv.INTER_LINEAR)
    return res


def warp_flow_open_cv_tensors(img_tensor, flow_tensor):
    def to_numpy(tensor):
        return tensor.detach().cpu().permute(1, 2, 0).numpy()

    return warp_flow_open_cv(to_numpy(img_tensor), to_numpy(flow_tensor))


def resample_bilinear(image, flow, debug=False, performance=False):
    if not performance:
        # in performance mode flow values are changed during warping, else flow is cloned
        flow = flow.clone()
    align_corners = True  # allign corners True creates behaviour like opencv.remap
    batch_size, _, h, w = image.size()
    affine_matrices = (
        torch.Tensor([[1, 0, 0], [0, 1, 0]])
        .unsqueeze(0)
        .expand(batch_size, -1, -1)
        .to(flow.device)
    )  # affine matrix for copy image as is, without transforming it
    grid_size = torch.Size((batch_size, 3, h, w))
    grid = torch.nn.functional.affine_grid(
        affine_matrices, grid_size, align_corners=align_corners
    )
    flow = flow.permute(0, 2, 3, 1)  # to [b, h, w, 2]
    flow[:, :, :, 0] *= 2.0 / max(w - 1, 1)
    flow[:, :, :, 1] *= 2.0 / max(h - 1, 1)
    grid += flow
    out_image = torch.nn.functional.grid_sample(
        image, grid, align_corners=align_corners
    )
    if debug:
        return {"output": out_image, "grid": grid}
    else:
        return out_image


try:
    from Forward_Warp import forward_warp, forward_warp_rescaled

    class forward_warp_permuted(forward_warp):
        def forward(self, im0, flow):
            assert len(im0.shape) == len(flow.shape) == 4
            assert im0.shape[0] == flow.shape[0]
            assert im0.shape[-2:] == flow.shape[-2:]
            assert flow.shape[1] == 2
            return super().forward(
                im0.contiguous(), flow.permute(0, 2, 3, 1).contiguous()
            )

    class forward_warp_rescaled_permuted(forward_warp_rescaled):
        def forward(self, im0, flow):
            assert len(im0.shape) == len(flow.shape) == 4
            assert im0.shape[0] == flow.shape[0]
            assert im0.shape[-2:] == flow.shape[-2:]
            assert flow.shape[1] == 2
            return super().forward(
                im0.contiguous(), flow.permute(0, 2, 3, 1).contiguous()
            )


except ModuleNotFoundError:
    pass


def invert_flow_batch(flow):
    device = flow.device
    inverse = torch.stack(
        [
            flow_instance.cpu().field().from_pixels().linverse().pixels().tensor()
            for flow_instance in flow
        ],
        dim=0,
    ).to(device=device)
    return inverse


class MaskCreator(torch.nn.Module):
    """Create a mask where the interesting stuff happens in the image.
    """

    def __init__(self, sigma: float, threshold: float) -> None:
        super().__init__()
        self.threshold = threshold
        kernel_size = 2 * int(np.ceil(3 * sigma)) + 1
        self.gaussian_filter = kornia.filters.GaussianBlur2d(
            (kernel_size,) * 2, (sigma,) * 2
        )

    def forward(
        self, forward_flow: torch.Tensor, backward_flow: torch.Tensor
    ) -> torch.Tensor:
        device = forward_flow.device
        forward_flow = torchfields.Field(forward_flow.cpu())
        backward_flow = torchfields.Field(backward_flow.cpu())
        summed = forward_flow.magnitude(keepdim=True) + backward_flow.magnitude(
            keepdim=True
        )
        summed = summed.to(device)
        blurred = self.gaussian_filter(summed)
        masked = blurred.ge(self.threshold).float()

        return masked


class MaskCreatorForward(MaskCreator):
    """Create a mask where the interesting stuff happens in the image.
    """

    def forward(self, forward_flow: torch.Tensor) -> torch.Tensor:
        assert len(forward_flow.shape) == 4

        backward_flow = invert_flow_batch(forward_flow)

        return super().forward(forward_flow, backward_flow)


def im2var(image):
    return torch.Tensor(image).permute(2, 0, 1).unsqueeze(0)


def create_flow_grid(flow, H, W):
    """create a grid equivalent to the input image with size H, W"""
    # Width and Height are different for thinking of images, or thinking of plain arrays...
    flow_grid = np.meshgrid(np.linspace(-1, 1, W), np.linspace(-1, 1, H))
    flow_grid = np.stack(flow_grid, axis=-1)

    # apply flow to grid
    # flow is pixel offset --> scale with image size
    flow_grid += flow / np.expand_dims(np.expand_dims((H, W), 0), 0)

    # move <H, W, 2> to <2, H, W>
    flow_grid = flow_grid.transpose(2, 0, 1)
    # create new Tensor, to have float32 for grid_sample function
    flow_grid = torch.Tensor(flow_grid).unsqueeze(0)  # [1, 2, H, W]
    flow_grid = flow_grid.transpose(1, 3)  # [1, W, H, 2]
    flow_grid = flow_grid.transpose(1, 2)  # [1, H, W, 2]
    # print("[DEBUG] create_flow_grid:", flow_grid.shape)
    return flow_grid


def warp_image_pt(image, flow):
    """warp an image by applying a flow grid"""
    if isinstance(image, torch.Tensor):
        H, W = image.shape[-2:]
    else:
        H, W = image.shape[:2]
    flow_grid = create_flow_grid(flow, H, W)
    # Actual warp
    image_warp = torch.nn.functional.grid_sample(
        im2var(image).float(), flow_grid, padding_mode="border", align_corners=True
    )
    return image_warp


def warp_image_pt_pwc(x, flo, debug=False):
    """
    warp an image/tensor (im2) back to im1, according to the optical flow
    x: [B, C, H, W] (im2)
    flo: [B, 2, H, W] flow
    """
    B, C, H, W = x.size()
    # mesh grid
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), 1).float()

    if x.is_cuda:
        grid = grid.cuda()
    vgrid = grid + flo

    # scale grid to [-1,1]
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0

    align_corners = True

    vgrid = vgrid.permute(0, 2, 3, 1)
    output = torch.nn.functional.grid_sample(x, vgrid, align_corners=align_corners)
    mask = torch.ones(x.size())
    if x.is_cuda:
        mask.cuda()
    mask = torch.nn.functional.grid_sample(mask, vgrid, align_corners=align_corners)

    # if W==128:
    # np.save('mask.npy', mask.cpu().data.numpy())
    # np.save('warp.npy', output.cpu().data.numpy())

    mask[mask < 0.9999] = 0
    mask[mask > 0] = 1

    if debug:
        return {"masked": output * mask, "output": output, "mask": mask, "grid": vgrid}
    else:
        return output * mask
