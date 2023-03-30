import torch


def mobius_addition(x, y, c):
    xy_prod = torch.sum(x * y, dim=-1).view(-1, 1)
    x_norm = torch.linalg.norm(x, dim=-1, ord=2).view(-1, 1)**2
    y_norm = torch.linalg.norm(y, dim=-1, ord=2).view(-1, 1)**2

    one_plus_2cxy = 1+2*c*xy_prod
    num = (one_plus_2cxy + c*y_norm)*x + (1-c*x_norm)*y
    den = one_plus_2cxy + c**2*x_norm*y_norm
    ret = num / den
    project(ret, c)
    return ret


def mobius_matrix_multiplication(M, x, c):
    batch_size = M.shape[0]
    # c can be negative when dealing with hyperspherical space (check GIE).
    sqrt = torch.sqrt(torch.abs(c))
    m_times_x = torch.bmm(M, x.view(batch_size, -1, 1)).view(batch_size, -1)
    mx_norm = torch.linalg.norm(m_times_x, dim=-1, ord=2).view(-1, 1)
    x_norm = torch.linalg.norm(x, dim=-1, ord=2).view(-1, 1)
    # In atanh(x), x \in (-1, 1).
    ret = 1/sqrt * torch.tanh(mx_norm/x_norm * torch.atanh(torch.clamp(sqrt*x_norm, min=-1 + 1e-10, max=1 - 1e-10))) * \
        m_times_x / mx_norm
    project(ret, c)
    return ret


def geodesic_dist(x, y, c):
    # c can be negative when dealing with hyperspherical space (check GIE).
    sqrt = torch.sqrt(torch.abs(c))
    # In atanh(x), x \in (-1, 1).
    sqrt_add_norm = torch.clamp(
        sqrt * torch.linalg.norm(mobius_addition(-x, y, c), dim=-1, ord=2).view(-1, 1), min=-1+1e-10, max=1-1e-10)
    return 2/sqrt * torch.atanh(sqrt_add_norm)


def distance(x, y):
    # In acosh(x), x \in [1, +inf).
    xy_norm = torch.linalg.norm(x-y, dim=-1, ord=2)**2
    x_norm = torch.linalg.norm(x, dim=-1, ord=2)**2
    y_norm = torch.linalg.norm(y, dim=-1, ord=2)**2
    delta = xy_norm/((1-x_norm) * (1-y_norm))
    return torch.acosh(torch.clamp(1 + 2 * delta, min=1))


def log_map(x, c):
    return map_(x, torch.tanh, c)


def exp_map(x, c):
    # In atanh(x), x \in (-1, 1).
    ret = map_(x, torch.atanh, c, clamp=lambda y: torch.clamp(y, min=-1 + 1e-10, max=1 - 1e-10))
    project(ret, c)
    return ret


def project(x, c):
    # Projects points to the Poincare ball.
    # In place!
    with torch.no_grad():
        # Compute L2-norms of x.
        x_norms = torch.linalg.norm(x, dim=-1, ord=2).view(-1, 1)
        # Compute limits. # c can be negative when dealing with hyperspherical space (check GIE).
        max_limits = 1-1e-10/torch.sqrt(torch.abs(c))
        x.data = torch.where(x_norms > max_limits, x * max_limits / x_norms, x)


def map_(x, f, c, clamp=None):
    x_norm = torch.linalg.norm(x, ord=2, dim=-1).view(-1, 1)
    # c can be negative when dealing with hyperspherical space (check GIE).
    sq_times_norm = torch.sqrt(torch.abs(c)) * x_norm
    if clamp is not None:
        sq_times_norm = clamp(sq_times_norm)
    return (f(sq_times_norm) / sq_times_norm) * x
