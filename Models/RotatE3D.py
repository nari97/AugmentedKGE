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
            self.create_embedding(self.dim, emb_type="entity", name="e_" + component)
            self.register_scale_constraint(emb_type="entity", name="e_" + component, p=2)
        for component in ['a', 'b', 'c', 'd']:
            self.create_embedding(self.dim, emb_type="relation", name="r_" + component)
        self.create_embedding(self.dim, emb_type="global", name="b")

    def _calc(self, h, r, t, b):
        (h_b, h_c, h_d) = h
        batch_size = h_b.shape[0]
        h_a = torch.zeros((batch_size, self.dim), device=h_b.device)
        (rc_a, rc_b, rc_c, rc_d) = QuaternionUtils.get_conjugate(r)
        rn = QuaternionUtils.quat_norm(r)**2

        (hr_a, hr_b, hr_c, hr_d) = QuaternionUtils.hamilton_product(
            QuaternionUtils.hamilton_product(r, (h_a, h_b, h_c, h_d)), (rc_a/rn, rc_b/rn, rc_c/rn, rc_d/rn))
        (t_b, t_c, t_d) = t

        # We add all components.
        scores = torch.zeros(batch_size, device=t_b.device)
        for c in [hr_b*b - t_b, hr_c*b - t_c, hr_d*b - t_d]:
            scores += torch.linalg.norm(c, dim=-1, ord=2)

        return -scores

    def return_score(self, is_predict=False):
        (head_emb, rel_emb, tail_emb) = self.current_batch

        (h_b, h_c, h_d) = (head_emb["e_b"], head_emb["e_c"], head_emb["e_d"])
        (t_b, t_c, t_d) = (tail_emb["e_b"], tail_emb["e_c"], tail_emb["e_d"])
        (r_a, r_b, r_c, r_d) = (rel_emb["r_a"], rel_emb["r_b"], rel_emb["r_c"], rel_emb["r_d"])
        b = self.current_global_embeddings["b"]

        return self._calc((h_b, h_c, h_d), (r_a, r_b, r_c, r_d), (t_b, t_c, t_d), b)
