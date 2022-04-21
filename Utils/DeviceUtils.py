import torch

use_gpu = False


def get_device():
    if use_gpu:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    return device
