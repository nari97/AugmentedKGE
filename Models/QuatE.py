import math
import torch
from Models.Model import Model


class QuatE(Model):
    def __init__(self, ent_total, rel_total, dims, use_gpu):
        """
        Args:
            ent_total (int): Total number of entities
            rel_total (int): Total number of relations
            dims (int): Number of dimensions for embeddings
        """
        super(QuatE, self).__init__(ent_total, rel_total, dims, "quate", use_gpu)

        for component in ['a', 'b', 'c', 'd']:
            self.create_embedding(self.dims, emb_type="entity", name="e_" + component, init='kaiming_uniform')
            self.create_embedding(self.dims, emb_type="relation", name="r_" + component, init='kaiming_uniform')

            self.register_scale_constraint(emb_type="entity", name="e_" + component, p=2)
            self.register_scale_constraint(emb_type="relation", name="r_" + component, p=2)

        # Special initialization: normalized quaternion with scalar component equal to zero.

        # Create embeddings but do not register.
        q_img_e = (self.create_embedding(self.dims, emb_type="entity", register=False, init=None).emb.data,
               self.create_embedding(self.dims, emb_type="entity", register=False, init='kaiming_uniform').emb.data,
               self.create_embedding(self.dims, emb_type="entity", register=False, init='kaiming_uniform').emb.data,
               self.create_embedding(self.dims, emb_type="entity", register=False, init='kaiming_uniform').emb.data)
        q_img_norm_e = self.quat_norm(q_img_e)

        q_img_r = (self.create_embedding(self.dims, emb_type="relation", register=False, init=None).emb.data,
               self.create_embedding(self.dims, emb_type="relation", register=False, init='kaiming_uniform').emb.data,
               self.create_embedding(self.dims, emb_type="relation", register=False, init='kaiming_uniform').emb.data,
               self.create_embedding(self.dims, emb_type="relation", register=False, init='kaiming_uniform').emb.data)
        q_img_norm_r = self.quat_norm(q_img_r)

        # Embeddings between -pi and pi.
        theta_e = self.create_embedding(self.dims, emb_type="entity", register=False,
                                        init='uniform', init_params=[-math.pi, math.pi]).emb.data
        theta_r = self.create_embedding(self.dims, emb_type="relation", register=False,
                                        init='uniform', init_params=[-math.pi, math.pi]).emb.data

        for idx, component in enumerate(['a', 'b', 'c', 'd']):
            e, r = self.get_embedding('entity', 'e_'+component), self.get_embedding('relation', 'r_'+component)
            if component is 'a':
                e.emb.data = e.emb.data * torch.cos(theta_e)
                r.emb.data = r.emb.data * torch.cos(theta_r)
            else:
                e.emb.data = e.emb.data * torch.sin(theta_e) * q_img_norm_e[idx]
                r.emb.data = r.emb.data * torch.sin(theta_r) * q_img_norm_r[idx]

    # Hamilton product
    def ham_prod(self, x_1, x_2):
        (a_1, b_1, c_1, d_1) = x_1
        (a_2, b_2, c_2, d_2) = x_2

        return (a_1 * a_2 - b_1 * b_2 - c_1 * c_2 - d_1 * d_2,
                a_1 * b_2 + b_1 * a_2 + c_1 * d_2 - d_1 * c_2,
                a_1 * c_2 - b_1 * d_2 + c_1 * a_2 + d_1 * b_2,
                a_1 * d_2 + b_1 * c_2 - c_1 * b_2 + d_1 * a_2)

    # Normalize quaternion
    def quat_norm(self, x):
        (x_a, x_b, x_c, x_d) = x
        den = torch.sqrt(torch.pow(x_a, 2) + torch.pow(x_b, 2) + torch.pow(x_c, 2) + torch.pow(x_d, 2))
        return x_a / den, x_b / den, x_c / den, x_d / den

    def _calc(self, h, r, t):
        # h * r_normalized
        (hr_a, hr_b, hr_c, hr_d) = self.ham_prod(h, self.quat_norm(r))
        (t_a, t_b, t_c, t_d) = t

        return torch.sum(hr_a * t_a, -1) + torch.sum(hr_b * t_b, -1) + \
            torch.sum(hr_c * t_c, -1) + torch.sum(hr_d * t_d, -1)

    def return_score(self, head_emb, rel_emb, tail_emb, is_predict=False):
        (h_a, h_b, h_c, h_d) = (head_emb["e_a"], head_emb["e_b"], head_emb["e_c"], head_emb["e_d"])
        (t_a, t_b, t_c, t_d) = (tail_emb["e_a"], tail_emb["e_b"], tail_emb["e_c"], tail_emb["e_d"])
        (r_a, r_b, r_c, r_d) = (rel_emb["r_a"], rel_emb["r_b"], rel_emb["r_c"], rel_emb["r_d"])

        return self._calc((h_a, h_b, h_c, h_d), (r_a, r_b, r_c, r_d), (t_a, t_b, t_c, t_d))
