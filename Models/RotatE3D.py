import math
import torch
from Models.Model import Model
from Utils import QuaternionUtils


class RotatE3D(Model):
    def __init__(self, ent_total, rel_total, dim):
        super(RotatE3D, self).__init__(ent_total, rel_total)
        self.dim = dim

    def get_default_loss(self):
        return 'soft_margin'

    def initialize_model(self):
        for component in ['b', 'c', 'd']:
            self.create_embedding(self.dim, emb_type="entity", name="e_" + component, reg=True)
        for component in ['theta', 'alpha', 'beta']:
            self.create_embedding(self.dim, emb_type="relation", name="r_" + component,
                                  init="uniform", init_params=[-math.pi, math.pi])
        self.create_embedding(self.dim, emb_type="global", name="b")

    def _calc(self, h, r, t, b):
        (h_b, h_c, h_d), (r_theta, r_alpha, r_beta) = h, r
        h_a = torch.zeros_like(h_b)

        stheta, salpha = torch.sin(r_theta/2), torch.sin(r_alpha)
        q = (torch.cos(r_theta/2), stheta * torch.cos(r_alpha), stheta * salpha * torch.cos(r_beta),
             stheta * salpha * torch.sin(r_beta))
        qs = QuaternionUtils.get_conjugate(q)

        (hr_a, hr_b, hr_c, hr_d) = QuaternionUtils.hamilton_product(
            QuaternionUtils.hamilton_product(q, (h_a, h_b, h_c, h_d)), qs)
        (t_b, t_c, t_d) = t

        return -torch.sum(QuaternionUtils.quat_norm((h_a, hr_b*b - t_b, hr_c*b - t_c, hr_d*b - t_d)), dim=-1)

    def return_score(self, is_predict=False):
        (head_emb, rel_emb, tail_emb) = self.current_batch

        (h_b, h_c, h_d) = (head_emb["e_b"], head_emb["e_c"], head_emb["e_d"])
        (t_b, t_c, t_d) = (tail_emb["e_b"], tail_emb["e_c"], tail_emb["e_d"])
        (r_theta, r_alpha, r_beta) = (rel_emb["r_theta"], rel_emb["r_alpha"], rel_emb["r_beta"])
        b = self.current_global_embeddings["b"]

        return self._calc((h_b, h_c, h_d), (r_theta, r_alpha, r_beta), (t_b, t_c, t_d), b)
