import numpy as np
import torch


def get_rotation_matrix(r, dim):
    # r = [r0, r1]
    # r as a block-diagonal matrix:
    #  cos r0, -sin r0,      0,       0
    #  sin r0,  cos r0,      0,       0
    #       0,       0, cos r1, -sin r1
    #       0,       0, sin r1,  cos r1
    # Even and odd indexes.
    even_indexes = torch.LongTensor(np.arange(0, dim, 2))
    odd_indexes = torch.LongTensor(np.arange(1, dim, 2))

    r_diag = torch.cos(r).repeat_interleave(2, dim=1)
    matrix = torch.diag_embed(r_diag)
    # Even rows, odd columns are -sin r.
    matrix[:, even_indexes, odd_indexes] = -torch.sin(r)
    # ODd rows, even columns are sin r.
    matrix[:, odd_indexes, even_indexes] = torch.sin(r)

    return matrix


def rotation_multiplication(r, e, dim):
    # r = [r0, r1]
    # r as a block-diagonal matrix:
    #  cos r0, -sin r0,      0,       0
    #  sin r0,  cos r0,      0,       0
    #       0,       0, cos r1, -sin r1
    #       0,       0, sin r1,  cos r1
    # e = [e0, e1, e2, e3]
    # r*e = [e0 cos r0 + e1 -sin r0, e1 cos r0 + e0 sin r0, e2 cos r1 + e3 -sin r1, e3 cos r1 + e2 sin r1]
    # r_diag = [cos r0, cos r0, cos r1, cos r1] (of the block-diagonal matrix)
    # r_diag*e gives all the first elements in r*e: [e0 cos r0, e1 cos r0, e2 cos r2, e3 cos r2] => diag_times_e
    # r_cdiag = [sin r0, sin r0, sin r1, sin r1] (of the block-diagonal matrix)
    # r_cdiag*e gives all the second elements in r*h with no sign and in different order:
    #   [e0 sin r0, e1 sin r0, e2 sin r1, e3 sin r1] => cdiag_times_e
    # diag_times_e[even] -= cdiag_times_e[odd] (the even positions of the result)
    # diag_times_e[odd] += cdiag_times_e[even] (the odd positions of the result)

    # Even and odd indexes.
    even_indexes = torch.LongTensor(np.arange(0, dim, 2))
    odd_indexes = torch.LongTensor(np.arange(1, dim, 2))

    r_diag = torch.cos(r).repeat_interleave(2, dim=1)
    diag_times_e = r_diag * e

    r_cdiag = torch.sin(r).repeat_interleave(2, dim=1)
    cdiag_times_e = r_cdiag * e

    diag_times_e[:, even_indexes] -= cdiag_times_e[:, odd_indexes]
    diag_times_e[:, odd_indexes] += cdiag_times_e[:, even_indexes]

    return diag_times_e


def reflection_multiplication(r, e, dim):
    # r = [r0, r1]
    # r as a block-diagonal matrix:
    #  cos r0,  sin r0,      0,       0
    #  sin r0, -cos r0,      0,       0
    #       0,       0, cos r1,  sin r1
    #       0,       0, sin r1, -cos r1
    # e = [e0, e1, e2, e3]
    # r*e = [e0 cos r0 + e1 sin r0, e1 -cos r0 + e0 sin r0, e2 cos r1 + e3 sin r1, e3 -cos r1 + e2 sin r1]
    # r_diag = [cos r0, cos r0, cos r2, cos r2] (of the block-diagonal matrix)
    # r_diag[even] *= -1
    # r_diag*e gives all the first elements in r*e: [e0 cos r0, e1 -cos r0, e2 cos r1, e3 -cos r1] => diag_times_e
    # r_cdiag = [sin r0, sin r0, sin r1, sin r1] (of the block-diagonal matrix)
    # r_cdiag*e gives all the second elements in r*h in different order:
    #   [e0 sin r0, e1 sin r0, e2 sin r1, e3 sin r1] => cdiag_times_e
    # diag_times_e[even] += cdiag_times_e[odd] (the even positions of the result)
    # diag_times_e[odd] += cdiag_times_e[even] (the odd positions of the result)

    # Even and odd indexes.
    even_indexes = torch.LongTensor(np.arange(0, dim, 2))
    odd_indexes = torch.LongTensor(np.arange(1, dim, 2))

    r_diag = torch.cos(r).repeat_interleave(2, dim=1)
    r_diag[:, even_indexes] *= -1
    diag_times_e = r_diag*e

    r_cdiag = torch.sin(r).repeat_interleave(2, dim=1)
    cdiag_times_e = r_cdiag*e

    diag_times_e[:, even_indexes] += cdiag_times_e[:, odd_indexes]
    diag_times_e[:, odd_indexes] += cdiag_times_e[:, even_indexes]

    return diag_times_e