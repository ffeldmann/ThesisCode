import torch

def numpy2torch(array):
    tensor = torch.from_numpy(array).float()  # torch.tensor(array, dtype=torch.float32)
    # tensor = tensor.permute(0, 3, 1, 2)
    return tensor

def torch2numpy(tensor):
    array = tensor.detach().cpu().numpy()
    # array = np.transpose(array, (0, 2, 3, 1))
    return array

def sure_to_numpy(tensor):
    if torch.is_tensor(tensor):
        return tensor.detach().cpu().numpy()
    elif type(tensor).__module__ != 'numpy':
        raise ValueError("Cannot convert {} to numpy array"
                         .format(type(tensor)))
    return tensor


def sure_to_torch(ndarray):
    """
    This function makes sure that the input is a torch tensor.
    You can use it even if you are not sure if your object is already a tensor or not.
    Args:
        ndarray: Probably array what you put inside.

    Returns: torch.tensor

    """
    if type(ndarray).__module__ == 'numpy':
        return torch.from_numpy(ndarray)
    elif not torch.is_tensor(ndarray):
        raise ValueError("Cannot convert {} to torch tensor"
                         .format(type(ndarray)))
    return ndarray