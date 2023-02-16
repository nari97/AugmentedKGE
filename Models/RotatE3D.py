import math
import torch
from Models.Model import Model
from Utils import QuaternionUtils


class RotatE3D(Model):
    """
    Chang Gao, Chengjie Sun, Lili Shan, Lei Lin, Mingjiang Wang: Rotate3D: Representing Relations as Rotations in
        Three-Dimensional Space for Knowledge Graph Embedding. CIKM 2020: 385-394.
    """
    def __init__(self, ent_total, rel_total, dim):
        """
            dim (int): Number of dimensions for embeddings
        """
        super(RotatE3D, self).__init__(ent_total, rel_total)
        self.dim = dim

    def get_default_loss(self):
        # Eq. (13).
        return 'soft_margin'

    def initialize_model(self):
        # The real part, a, is zero for entity embeddings (see above Eq. (9)).
        for component in ['b', 'c', 'd']:
            self.create_embedding(self.dim, emb_type="entity", name="e_" + component, reg=True)
        # See Section 5.3.
        # From the paper: "...are uniformly initialized between −pi and pi. This guarantees that u is a unit vector."
        # We are also adding rescaling to make sure they are always between -pi and pi.
        for component in ['theta', 'alpha', 'beta']:
            self.create_embedding(self.dim, emb_type="relation", name="r_" + component,
                                  init_method="uniform", init_params=[-math.pi, math.pi],
                                  norm_method="rescaling", norm_params={"a": -math.pi, "b": math.pi})
        # Bias (see below Eq. (9)).
        # From the paper: "Relation-specific biases are initialized to 1."
        self.create_embedding(self.dim, emb_type="global", name="b", init_method="uniform", init_params=[1, 1])

    def _calc(self, h, r, t, b):
        (h_b, h_c, h_d), (r_theta, r_alpha, r_beta) = h, r
        # a is just zeros.
        h_a = torch.zeros_like(h_b)
        # Auxiliary used a couple of times.
        sin_theta_by_two, sin_alpha = torch.sin(r_theta/2), torch.sin(r_alpha)
        # See above Eq. (9) and Section 5.3.
        q = (torch.cos(r_theta/2), sin_theta_by_two * torch.cos(r_alpha), sin_theta_by_two * sin_alpha * torch.cos(r_beta),
             sin_theta_by_two * sin_alpha * torch.sin(r_beta))
        # Get conjugate.
        qc = QuaternionUtils.get_conjugate(q)
        # Eq. (8).
        (hr_a, hr_b, hr_c, hr_d) = QuaternionUtils.hamilton_product(
            QuaternionUtils.hamilton_product(q, (h_a, h_b, h_c, h_d)), qc)
        (t_b, t_c, t_d) = t
        # Eq. (11).
        return -torch.sum(QuaternionUtils.quat_norm((hr_a*b, hr_b*b - t_b, hr_c*b - t_c, hr_d*b - t_d)), dim=-1)

    def return_score(self, is_predict=False):
        (head_emb, rel_emb, tail_emb) = self.current_batch

        (h_b, h_c, h_d) = (head_emb["e_b"], head_emb["e_c"], head_emb["e_d"])
        (t_b, t_c, t_d) = (tail_emb["e_b"], tail_emb["e_c"], tail_emb["e_d"])
        (r_theta, r_alpha, r_beta) = (rel_emb["r_theta"], rel_emb["r_alpha"], rel_emb["r_beta"])
        b = self.current_global_embeddings["b"]

        return self._calc((h_b, h_c, h_d), (r_theta, r_alpha, r_beta), (t_b, t_c, t_d), b)
