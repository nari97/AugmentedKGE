import math
import torch


# Since we train in the tangent space and then switch to Poincare, the norms may not be <1, so we enforce that.
def mobius_addition(x, y, c):
    xy_prod = torch.sum(x * y, dim=-1).view(-1, 1)
    x_norm = torch.clamp(torch.linalg.norm(x, dim=-1, ord=2).view(-1, 1), max=1-1e-10)**2
    y_norm = torch.clamp(torch.linalg.norm(y, dim=-1, ord=2).view(-1, 1), max=1-1e-10)**2

    num = (1+2*c*xy_prod + c*y_norm)*x + (1-c*x_norm)*y
    den = 1+2*c*xy_prod + c**2*x_norm * y_norm
    return num / den


def geodesic_dist(x, y, c):
    add_norm = torch.clamp(torch.linalg.norm(mobius_addition(-x, y, c), dim=-1, ord=2), max=1-1e-10)
    return 2/math.sqrt(c) * torch.atanh(math.sqrt(c)*add_norm)


def distance(x, y):
    xy_norm = torch.clamp(torch.linalg.norm(x-y, dim=-1, ord=2), max=1-1e-10)**2
    x_norm = torch.clamp(torch.linalg.norm(x, dim=-1, ord=2), max=1-1e-10)**2
    y_norm = torch.clamp(torch.linalg.norm(y, dim=-1, ord=2), max=1-1e-10)**2
    return torch.acosh(1 + 2 * xy_norm/((1-x_norm) * (1-y_norm)))

def log_map(x, c):
    return map_(x, torch.tanh, c)


def exp_map(x, c):
    return map_(x, torch.atanh, c)


def map_(x, f, c):
    x_norm = torch.clamp(torch.linalg.norm(x, ord=2, dim=-1), max=1-1e-10)
    scalar = f(math.sqrt(c) * x_norm) / (math.sqrt(c) * x_norm)
    return scalar.view(-1, 1) * x
