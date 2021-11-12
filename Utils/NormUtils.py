import torch
def clamp_norm(input, p = 2, dim = -1, maxnorm = 1):
    """
        Computed Ln norm and clamps value of input to max

        Args:
            input (Tensor): Tensor containing vector of data to be clamped
            p (int) : L1 or L2 norm. Default: 2
            dim (int) : Dimension across which norm is to be calculated. Default: -1
            maxnorm (int) : Maximum value after which data is clamped. Default: 1

        Returns:
            ans: Tensor that has been normalised and has had values clamped
    """

    norm = torch.norm(input, p = p, dim = dim, keepdim=True)
    mask = (norm<maxnorm).long()
    ans = mask*input + (1-mask)*(input/torch.clamp_min(norm, 10e-8)*maxnorm)
    return ans

def normalize(input, p = 2, dim = -1):
    """
        Computes Ln norm

        Args:
            input (Tensor): Tensor containing vector of data to be normalized
            p (int) : L1 or L2 norm. Default: 2
            dim (int) : Dimension across which norm is to be calculated. Default: -1
            

        Returns:
            ans: Tensor that has been normalised
    """

    ans = torch.nn.functional.normalize(input, p = p, dim = dim)
    return ans