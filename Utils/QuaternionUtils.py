import torch


# Hamilton product
def hamilton_product(x_1, x_2):
    (a_1, b_1, c_1, d_1) = x_1
    (a_2, b_2, c_2, d_2) = x_2

    return (a_1 * a_2 - b_1 * b_2 - c_1 * c_2 - d_1 * d_2,
            a_1 * b_2 + b_1 * a_2 + c_1 * d_2 - d_1 * c_2,
            a_1 * c_2 - b_1 * d_2 + c_1 * a_2 + d_1 * b_2,
            a_1 * d_2 + b_1 * c_2 - c_1 * b_2 + d_1 * a_2)


def addition(x, y):
    (x_a, x_b, x_c, x_d) = x
    (y_a, y_b, y_c, y_d) = y
    return x_a + y_a, x_b + y_b, x_c + y_c, x_d + y_d


def get_conjugate(x):
    (x_a, x_b, x_c, x_d) = x
    return x_a, -x_b, -x_c, -x_d


def quat_norm_square(x):
    (x_a, x_b, x_c, x_d) = x
    return torch.pow(x_a, 2) + torch.pow(x_b, 2) + torch.pow(x_c, 2) + torch.pow(x_d, 2)


def quat_norm(x):
    return torch.sqrt(quat_norm_square(x))


def normalize_quaternion(x):
    (x_a, x_b, x_c, x_d) = x
    den = quat_norm(x)
    return x_a / den, x_b / den, x_c / den, x_d / den


def inner_product(x, y):
    (x_a, x_b, x_c, x_d) = x
    (y_a, y_b, y_c, y_d) = y
    return torch.sum(x_a * y_a, -1) + torch.sum(x_b * y_b, -1) + torch.sum(x_c * y_c, -1) + torch.sum(x_d * y_d, -1)


# Get the three angles of a quaternion.
# Check Eq. (10) in the DensE paper (https://arxiv.org/pdf/2008.04548v2.pdf).
def get_angles(x):
    norm = quat_norm(x)
    return get_angles_with_norm(x, norm)


def get_angles_with_norm(x, norm):
    (x_a, x_b, x_c, x_d) = x

    # In acos(x), x \in (-1, 1).
    psi = torch.acos(torch.clamp(x_a / norm, min=-1 + 1e-10, max=1 - 1e-10)) * 2
    theta = torch.acos(torch.clamp(x_d / (norm * torch.sin(psi / 2)), min=-1 + 1e-10, max=1 - 1e-10))
    phi = torch.acos(torch.clamp(x_b / (norm * torch.sin(psi / 2) * torch.sin(theta)), min=-1 + 1e-10, max=1 - 1e-10))

    return psi, theta, phi


# Get the inverse of the quaternion (https://www.mathworks.com/help/aeroblks/quaternioninverse.html).
def inverse(x):
    norm_sq = quat_norm_square(x)
    return inverse_with_norm(x, norm_sq)


# Computes the inverse assuming the squared norm has been computed previously.
def inverse_with_norm(x, norm_sq):
    (x_a, x_b, x_c, x_d) = x
    return x_a/norm_sq, -x_b/norm_sq, -x_c/norm_sq, -x_d/norm_sq
