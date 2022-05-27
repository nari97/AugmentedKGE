import math
import torch


def mobius_addition(x, y, c):
    xy_prod = torch.sum(x * y, dim=-1).view(-1, 1)
    x_norm = torch.linalg.norm(x, dim=-1, ord=2).view(-1, 1)**2
    y_norm = torch.linalg.norm(y, dim=-1, ord=2).view(-1, 1)**2

    num = (1+2*c*xy_prod + c*y_norm)*x + (1-c*x_norm)*y
    den = 1+2*c*xy_prod + c**2*x_norm * y_norm
    ret = num / den

    if True in torch.isnan(ret):
        for i, y in enumerate(torch.isnan(ret)):
            if y.item() is True:
                print('NaN!')

    return ret


def geodesic_dist(x, y, c):
    add_norm = torch.clamp(torch.linalg.norm(mobius_addition(-x, y, c), dim=-1, ord=2), min=1e-10, max=1-1e-10)
    #add_norm = torch.linalg.norm(mobius_addition(-x, y, c), dim=-1, ord=2)
    ret = 2/math.sqrt(c) * torch.atanh(math.sqrt(c)*add_norm)

    if True in torch.isnan(ret):
        for i, y in enumerate(torch.isnan(ret)):
            if y.item() is True:
                print('NaN!')

    return ret


def log_map(x, c):
    return map_(x, torch.tanh, c)


def exp_map(x, c):
    return map_(x, torch.atanh, c)


def map_(x, f, c):
    x_norm = torch.clamp(torch.linalg.norm(x, ord=2, dim=-1), min=1e-10, max=1-1e-10)
    #x_norm = torch.linalg.norm(x, ord=2, dim=-1)
    scalar = f(math.sqrt(c) * x_norm) / (math.sqrt(c) * x_norm)

    if True in torch.isnan(scalar):
        for i, y in enumerate(torch.isnan(scalar)):
            if y.item() is True:
                print('NaN!')

    return scalar.view(-1, 1) * x
