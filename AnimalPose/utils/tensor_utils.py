import torch

def numpy2torch(array):
    tensor = torch.from_numpy(array).float()  # torch.tensor(array, dtype=torch.float32)
    # tensor = tensor.permute(0, 3, 1, 2)
    return tensor

def torch2numpy(tensor):
    array = tensor.detach().cpu().numpy()
    # array = np.transpose(array, (0, 2, 3, 1))
    return array