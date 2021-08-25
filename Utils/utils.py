import torch

def clamp_norm(input, p = 2, dim = -1, maxnorm = 1):

    norm = torch.norm(input, p = p, dim = dim, keepdim=True)
    mask = (norm<maxnorm).long()

    ans = mask*input + (1-mask)*(input/torch.clamp_min_(norm, 10e-8)*maxnorm)

    return ans