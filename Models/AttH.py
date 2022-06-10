import math
import numpy as np
import torch
from Models.AttE import AttE
from Utils import PoincareUtils


# https://github.com/tensorflow/neural-structured-learning/blob/master/research/kg_hyp_emb/models/hyperbolic.py#L116
class AttH(AttE):
    def __init__(self, ent_total, rel_total, dim):
        super(AttH, self).__init__(ent_total, rel_total, dim)

    # This requires a special SGD: https://github.com/ibalazevic/multirelational-poincare/blob/master/rsgd.py
    # Instead, we will optimize the tangent space, then, we map them back to the Poincare ball.
    # Check A.4 in https://arxiv.org/pdf/2005.00545.pdf
    def _calc_predict(self, h, bh, r, r_theta, r_phi, ar, t, bt):
        batch_size = h.shape[0]

        # Map h, r and t from Euclidean to Poincare.
        h, r, t = PoincareUtils.exp_map(h, self.c), PoincareUtils.exp_map(r, self.c), PoincareUtils.exp_map(t, self.c)

        # Rotate and reflect.
        h_rot = torch.bmm(self.get_matrix(r_theta, batch_size, 'rot'), h.view(-1, self.dim, 1)).view(-1, self.dim)
        h_ref = torch.bmm(self.get_matrix(r_phi, batch_size, 'ref'), h.view(-1, self.dim, 1)).view(-1, self.dim)

        # Attention.
        alpha = torch.nn.functional.softmax(torch.cat((torch.sum(ar * h_rot, dim=-1).view(-1, 1),
                                                       torch.sum(ar * h_ref, dim=-1).view(-1, 1)), dim=1), dim=1)
        att = alpha[:, 0].view(-1, 1) * h_rot + alpha[:, 1].view(-1, 1) * h_ref

        hr = PoincareUtils.mobius_addition(att, r, self.c)

        return -PoincareUtils.geodesic_dist(hr, r, self.c) ** 2 + bh.flatten() + bt.flatten()
