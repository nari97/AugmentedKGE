import math
import numpy as np
import torch
from Models.Model import Model


# https://github.com/tensorflow/neural-structured-learning/blob/master/research/kg_hyp_emb/models/hyperbolic.py#L116
class AttE(Model):
    def __init__(self, ent_total, rel_total, dim):
        super(AttE, self).__init__(ent_total, rel_total)
        self.dim = dim

    def get_default_loss(self):
        return 'soft'

    def initialize_model(self):
        self.create_embedding(self.dim, emb_type="entity", name="e")
        self.create_embedding(self.dim, emb_type="relation", name="r")
        self.create_embedding(1, emb_type="entity", name="b")
        self.create_embedding(1, emb_type="relation", name="a")
        # Note: we do not include c as we are not learning in the Poincare space.
        self.c = 1

        # They are regularized in the implementation; check:
        # https://github.com/tensorflow/neural-structured-learning/blob/master/research/kg_hyp_emb/models/euclidean.py#L145
        self.create_embedding(math.floor(self.dim/2), emb_type="relation", name="r_theta",
                              init="uniform", init_params=[0, 2 * math.pi], reg=True)
        self.create_embedding(math.floor(self.dim/2), emb_type="relation", name="r_phi",
                              init="uniform", init_params=[0, 2 * math.pi], reg=True)

    def get_matrix(self, r, batch_size, optype):
        # r = [a, b]
        # r as a block-diagonal matrix using rotation:
        #   cos(a), -sin(a), 0, 0, 0
        #   sin(a), cos(a), 0, 0, 0
        #   0, 0, cos(b), -sin(b), 0
        #   0, 0, sin(b), cos(b), 0
        #   0, 0, 0, 0, 0
        # r as a block-diagonal matrix using reflection:
        #   cos(a), sin(a), 0, 0, 0
        #   sin(a), -cos(a), 0, 0, 0
        #   0, 0, cos(b), sin(b), 0
        #   0, 0, sin(b), -cos(b), 0
        #   0, 0, 0, 0, 0
        even_indexes = torch.LongTensor(np.arange(0, self.dim - 1 if self.dim % 2 == 1 else self.dim, 2))
        odd_indexes = torch.LongTensor(np.arange(1, self.dim - 1 if self.dim % 2 == 1 else self.dim, 2))
        r_diag = torch.cos(r.repeat_interleave(2, dim=1))

        if r_diag.shape[1] != self.dim:
            r_diag = torch.cat((r_diag, torch.zeros(batch_size, 1)), dim=1)

        if optype is 'ref':
            # Make odd indexes negative.
            r_diag[:,odd_indexes] *= -1

        r_u = torch.sin(r.repeat_interleave(2, dim=1))
        # Make odd indexes zero.
        r_u[:, odd_indexes] *= 0

        if optype is 'rot':
            # Make even indexes negative.
            r_u[:,even_indexes] *= -1

        r_l = torch.sin(r.repeat_interleave(2, dim=1))
        # Make odd indexes zero.
        r_l[:, odd_indexes] *= 0

        return torch.diag_embed(r_diag) + torch.diag_embed(r_u[:,:self.dim - 1], offset=1) + \
              torch.diag_embed(r_l[:,:self.dim - 1], offset=-1)

    # In this case, the predict and train are the same. They are not in AttH.
    def _calc_predict(self, h, bh, r, r_theta, r_phi, ar, t, bt):
        return self._calc_train(h, bh, r, r_theta, r_phi, ar, t, bt)

    def _calc_train(self, h, bh, r, r_theta, r_phi, ar, t, bt):
        batch_size = h.shape[0]

        # Rotate and reflect.
        h_rot = torch.bmm(self.get_matrix(r_theta, batch_size, 'rot'), h.view(-1, self.dim, 1)).view(-1, self.dim)
        h_ref = torch.bmm(self.get_matrix(r_phi, batch_size, 'ref'), h.view(-1, self.dim, 1)).view(-1, self.dim)

        # Attention.
        alpha = torch.nn.functional.softmax(torch.cat((torch.sum(ar * h_rot, dim=-1).view(-1, 1),
                                                       torch.sum(ar * h_ref, dim=-1).view(-1, 1)), dim=1), dim=1)
        att = alpha[:,0].view(-1, 1)*h_rot + alpha[:,1].view(-1, 1)*h_ref

        return -torch.linalg.norm(att + r - t, dim=-1, ord=2)**2 + bh.flatten() + bt.flatten()

    def return_score(self, is_predict=False):
        (head_emb, rel_emb, tail_emb) = self.current_batch

        h, bh = head_emb["e"], head_emb["b"]
        t, bt = tail_emb["e"], tail_emb["b"]
        r, r_theta, r_phi, ar = rel_emb["r"], rel_emb["r_theta"], rel_emb["r_phi"], rel_emb["a"]

        if not is_predict:
            return self._calc_train(h, bh, r, r_theta, r_phi, ar, t, bt)
        else:
            return self._calc_predict(h, bh, r, r_theta, r_phi, ar, t, bt)
