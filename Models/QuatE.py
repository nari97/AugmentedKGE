import math
import torch
from Models.Model import Model
from Utils import QuaternionUtils


class QuatE(Model):
    def __init__(self, ent_total, rel_total, dim):
        super(QuatE, self).__init__(ent_total, rel_total)
        self.dim = dim

    def get_default_loss(self):
        return 'soft'

    def initialize_model(self):
        for component in ['a', 'b', 'c', 'd']:
            self.create_embedding(self.dim, emb_type="entity", name="e_" + component, init='kaiming_uniform', reg=True)
            self.create_embedding(self.dim, emb_type="relation", name="r_" + component, init='kaiming_uniform', reg=True)

        # Special initialization: normalized quaternion with scalar component equal to zero.

        # Create embeddings but do not register.
        q_img_e = (self.create_embedding(self.dim, emb_type="entity", register=False, init=None).emb.data,
               self.create_embedding(self.dim, emb_type="entity", register=False, init='kaiming_uniform').emb.data,
               self.create_embedding(self.dim, emb_type="entity", register=False, init='kaiming_uniform').emb.data,
               self.create_embedding(self.dim, emb_type="entity", register=False, init='kaiming_uniform').emb.data)
        q_img_norm_e = QuaternionUtils.quat_norm(q_img_e)

        q_img_r = (self.create_embedding(self.dim, emb_type="relation", register=False, init=None).emb.data,
               self.create_embedding(self.dim, emb_type="relation", register=False, init='kaiming_uniform').emb.data,
               self.create_embedding(self.dim, emb_type="relation", register=False, init='kaiming_uniform').emb.data,
               self.create_embedding(self.dim, emb_type="relation", register=False, init='kaiming_uniform').emb.data)
        q_img_norm_r = QuaternionUtils.quat_norm(q_img_r)

        # Embeddings between -pi and pi.
        theta_e = self.create_embedding(self.dim, emb_type="entity", register=False,
                                        init='uniform', init_params=[-math.pi, math.pi]).emb.data
        theta_r = self.create_embedding(self.dim, emb_type="relation", register=False,
                                        init='uniform', init_params=[-math.pi, math.pi]).emb.data

        for idx, component in enumerate(['a', 'b', 'c', 'd']):
            e, r = self.get_embedding('entity', 'e_'+component), self.get_embedding('relation', 'r_'+component)
            if component is 'a':
                e.emb.data *= torch.cos(theta_e)
                r.emb.data *= torch.cos(theta_r)
            else:
                e.emb.data *= torch.sin(theta_e) * q_img_norm_e[idx]
                r.emb.data *= torch.sin(theta_r) * q_img_norm_r[idx]

    def _calc(self, h, r, t):
        return QuaternionUtils.inner_product(
            QuaternionUtils.hamilton_product(h, QuaternionUtils.normalize_quaternion(r)), t)

    def return_score(self, is_predict=False):
        (head_emb, rel_emb, tail_emb) = self.current_batch

        (h_a, h_b, h_c, h_d) = (head_emb["e_a"], head_emb["e_b"], head_emb["e_c"], head_emb["e_d"])
        (t_a, t_b, t_c, t_d) = (tail_emb["e_a"], tail_emb["e_b"], tail_emb["e_c"], tail_emb["e_d"])
        (r_a, r_b, r_c, r_d) = (rel_emb["r_a"], rel_emb["r_b"], rel_emb["r_c"], rel_emb["r_d"])

        return self._calc((h_a, h_b, h_c, h_d), (r_a, r_b, r_c, r_d), (t_a, t_b, t_c, t_d))
