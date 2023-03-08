import math
import torch
from Models.Model import Model
from Utils import QuaternionUtils


class QuatE(Model):
    """
    Shuai Zhang, Yi Tay, Lina Yao, Qi Liu: Quaternion Knowledge Graph Embeddings. NeurIPS 2019: 2731-2741.
    """
    def __init__(self, ent_total, rel_total, dim):
        """
            dim (int): Number of dimensions for embeddings
        """
        super(QuatE, self).__init__(ent_total, rel_total)
        self.dim = dim

    def get_default_loss(self):
        # Eq. (7).
        return 'soft'

    def get_score_sign(self):
        # The paper uses soft loss, so that means positive scores will be larger than negative scores.
        return 1

    def initialize_model(self):
        # See paragraph "Quaternion Embeddings of Knowledge Graphs" for embeddings and Eq. (7). for regularization.
        for component in ['a', 'b', 'c', 'd']:
            self.create_embedding(self.dim, emb_type="entity", name="e_" + component, init_method=None, reg=True)
            self.create_embedding(self.dim, emb_type="relation", name="r_" + component, init_method=None, reg=True)

        # Special initialization: normalized quaternion with scalar component equal to zero.
        # See "Initialization" paragraph and Eq. (8).

        # Two normalized quaternions with scalar part equal to zero. Create embeddings but do not register.
        q_img_e = (torch.zeros_like(self.get_embedding('entity', 'e_a').emb),
                   self.create_embedding(self.dim, emb_type="entity", register=False,
                                         init_method='kaiming_uniform').emb.data,
                   self.create_embedding(self.dim, emb_type="entity", register=False,
                                         init_method='kaiming_uniform').emb.data,
                   self.create_embedding(self.dim, emb_type="entity", register=False,
                                         init_method='kaiming_uniform').emb.data)
        q_img_norm_e = QuaternionUtils.quat_norm(q_img_e)

        q_img_r = (torch.zeros_like(self.get_embedding('relation', 'r_a').emb),
                   self.create_embedding(self.dim, emb_type="relation", register=False,
                                         init_method='kaiming_uniform').emb.data,
                   self.create_embedding(self.dim, emb_type="relation", register=False,
                                         init_method='kaiming_uniform').emb.data,
                   self.create_embedding(self.dim, emb_type="relation", register=False,
                                         init_method='kaiming_uniform').emb.data)
        q_img_norm_r = QuaternionUtils.quat_norm(q_img_r)

        phi = self.create_embedding(1, emb_type="global", register=False, name="phi", init_method='xavier_normal').emb

        # Embeddings between -pi and pi.
        theta_e = self.create_embedding(self.dim, emb_type="entity", register=False,
                                        init_method='uniform', init_params=[-math.pi, math.pi]).emb.data
        theta_r = self.create_embedding(self.dim, emb_type="relation", register=False,
                                        init_method='uniform', init_params=[-math.pi, math.pi]).emb.data

        # Eq. (8).
        for idx, component in enumerate(['a', 'b', 'c', 'd']):
            e, r = self.get_embedding('entity', 'e_' + component), self.get_embedding('relation', 'r_' + component)
            if component == 'a':
                e.emb.data = phi * torch.cos(theta_e)
                r.emb.data = phi * torch.cos(theta_r)
            else:
                e.emb.data = phi * torch.sin(theta_e) * q_img_norm_e[idx]
                r.emb.data = phi * torch.sin(theta_r) * q_img_norm_r[idx]

    def _calc(self, h, r, t):
        # Eqs. (5) and (6).
        # From the paper: "Note that the loss function is in Euclidean space, as we take the summation of all components
        #   when computing the scoring function in Equation (6)."
        return QuaternionUtils.inner_product(
            QuaternionUtils.hamilton_product(h, QuaternionUtils.normalize_quaternion(r)), t)

    def return_score(self, is_predict=False):
        (head_emb, rel_emb, tail_emb) = self.current_batch

        h = (head_emb["e_a"], head_emb["e_b"], head_emb["e_c"], head_emb["e_d"])
        t = (tail_emb["e_a"], tail_emb["e_b"], tail_emb["e_c"], tail_emb["e_d"])
        r = (rel_emb["r_a"], rel_emb["r_b"], rel_emb["r_c"], rel_emb["r_d"])

        return self._calc(h, r, t)
