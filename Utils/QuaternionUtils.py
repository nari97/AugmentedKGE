import torch


# Hamilton product
def hamilton_product(x_1, x_2):
    (a_1, b_1, c_1, d_1) = x_1
    (a_2, b_2, c_2, d_2) = x_2

    return (a_1 * a_2 - b_1 * b_2 - c_1 * c_2 - d_1 * d_2,
            a_1 * b_2 + b_1 * a_2 + c_1 * d_2 - d_1 * c_2,
            a_1 * c_2 - b_1 * d_2 + c_1 * a_2 + d_1 * b_2,
            a_1 * d_2 + b_1 * c_2 - c_1 * b_2 + d_1 * a_2)


def get_conjugate(x):
    (x_a, x_b, x_c, x_d) = x
    return x_a, -x_b, -x_c, -x_d


def quat_norm(x):
    (x_a, x_b, x_c, x_d) = x
    return torch.sqrt(torch.pow(x_a, 2) + torch.pow(x_b, 2) + torch.pow(x_c, 2) + torch.pow(x_d, 2))


def normalize_quaternion(x):
    (x_a, x_b, x_c, x_d) = x
    den = quat_norm(x)
    return x_a / den, x_b / den, x_c / den, x_d / den


def inner_product(x, y):
    (x_a, x_b, x_c, x_d) = x
    (y_a, y_b, y_c, y_d) = y
    return torch.sum(x_a * y_a, -1) + torch.sum(x_b * y_b, -1) + torch.sum(x_c * y_c, -1) + torch.sum(x_d * y_d, -1)


# Get the three angles of the unit quaternion.
# Check Eq. 10: https://arxiv.org/pdf/2008.04548v2.pdf
def get_angles(x, inv=False):
    (x_a, x_b, x_c, x_d) = x
    norm = quat_norm(x)

    psi = torch.acos(x_a/norm) * 2
    theta = torch.acos(x_d/(norm * (-1 if inv else 1)*torch.sin(psi/2)))
    phi = torch.acos(x_b/(norm * (-1 if inv else 1)*torch.sin(psi/2) * torch.sin(theta)))

    return psi, theta, phi
