import math
import torch
from Models.Model import Model
from Utils import QuaternionUtils


class DensE(Model):
    def __init__(self, ent_total, rel_total, dim):
        super(DensE, self).__init__(ent_total, rel_total)
        self.dim = dim

    def get_default_loss(self):
        return 'soft_margin'

    def initialize_model(self):
        for component in ['x', 'y', 'z']:
            self.create_embedding(self.dim, emb_type="entity", name="e_" + component)

        for component in ['a', 'b', 'c', 'd']:
            self.create_embedding(self.dim, emb_type="relation", name="r_" + component)

    # This is the matrix in Eq. 8
    def r_q(self, r, x, inv=False):
        r_norm = QuaternionUtils.quat_norm(r)
        (psi, theta, phi) = QuaternionUtils.get_angles(r, inv)

        c, s = torch.cos(psi), torch.sin(psi)
        v_x = torch.sin(theta) * torch.cos(phi)
        v_y = torch.sin(theta) * torch.sin(phi)
        v_z = torch.cos(theta)

        def prod(p1, p2):
            (p1_x, p1_y, p1_z) = p1
            (p2_x, p2_y, p2_z) = p2
            return p1_x*p2_x + p1_y*p2_y + p1_z*p2_z

        ret_x = prod((c+torch.pow(v_x, 2)*(1-c), v_x*v_y*(1-c)+v_z*s, v_x*v_z*(1-c)-v_y*s), x)
        ret_y = prod((v_x*v_y*(1-c)-v_z*s, c+torch.pow(v_y, 2)*(1-c), v_y*v_z*(1-c)+v_x*s), x)
        ret_z = prod((v_x*v_z*(1-c)+v_y*s, v_y*v_z*(1-c)-v_x*s, c+torch.pow(v_z, 2)*(1-c)), x)

        return (r_norm*ret_x, r_norm*ret_y, r_norm*ret_z)

    def _calc(self, h, r, t):
        (h_x, h_y, h_z) = h
        (t_x, t_y, t_z) = t

        batch_size = h_x.shape[0]

        (h_prod_x, h_prod_y, h_prod_z) = self.r_q(r, h)
        (t_prod_x, t_prod_y, t_prod_z) = self.r_q(r, t, inv=True)

        # https://github.com/anonymous-dense-submission/DensE/blob/master/codes/model.py#L296
        # We add all components.
        scores = torch.zeros(batch_size, device=h_x.device)
        for c in [h_prod_x-t_x, h_prod_y-t_y, h_prod_z-t_z, t_prod_x-h_x, t_prod_y-h_y, t_prod_z-h_z]:
            scores += torch.linalg.norm(c, dim=-1, ord=2)

        return -.5 * scores

    def return_score(self, is_predict=False):
        (head_emb, rel_emb, tail_emb) = self.current_batch

        (h_x, h_y, h_z) = (head_emb["e_x"], head_emb["e_y"], head_emb["e_z"])
        (t_x, t_y, t_z) = (tail_emb["e_x"], tail_emb["e_y"], tail_emb["e_z"])
        (r_a, r_b, r_c, r_d) = (rel_emb["r_a"], rel_emb["r_b"], rel_emb["r_c"], rel_emb["r_d"])

        return self._calc((h_x, h_y, h_z), (r_a, r_b, r_c, r_d), (t_x, t_y, t_z))
