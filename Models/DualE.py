import math
import torch
from Models.Model import Model
from Utils import QuaternionUtils


class DualE(Model):
    """
    Zongsheng Cao, Qianqian Xu, Zhiyong Yang, Xiaochun Cao, Qingming Huang: Dual Quaternion Knowledge Graph Embeddings.
        AAAI 2021: 6894-6902.
    """
    def __init__(self, ent_total, rel_total, dim, variant='full'):
        """
            dim (int): Number of dimensions for embeddings
            variant can be either cross, in which d' is obtained by normalization (see 'Ablation Study on Dual
                Quaternion Normalization' in the experiments), or full.
        """
        super(DualE, self).__init__(ent_total, rel_total)
        self.dim = dim
        self.variant = variant

    def get_default_loss(self):
        # Eq. (9).
        return 'soft'

    def get_score_sign(self):
        # The paper uses soft loss, so that means positive scores will be larger than negative scores.
        return 1

    def initialize_model(self):
        # Eq. (9) includes regularization of the embeddings.
        for c in ['a', 'b', 'c', 'd']:
            for n in ['0', '1']:
                self.create_embedding(self.dim, emb_type="entity", name="e_"+c+n, init_method=None, reg=True)
                self.create_embedding(self.dim, emb_type="relation", name="r_"+c+n, init_method=None, reg=True)

        # Special initialization in appendix:
        #   https://github.com/Lion-ZS/DualE/blob/main/DualE-master/Supplement%20of%20DualE/Supplement%20of%20DualE.pdf
        def init_quat(etype, prefix):
            q = (torch.zeros_like(self.get_embedding(etype, prefix+'_a0').emb),
                 self.create_embedding(self.dim, emb_type=etype, register=False,
                                       init_method='uniform', init_params=[0, 1]).emb.data,
                 self.create_embedding(self.dim, emb_type=etype, register=False,
                                       init_method='uniform', init_params=[0, 1]).emb.data,
                 self.create_embedding(self.dim, emb_type=etype, register=False,
                                       init_method='uniform', init_params=[0, 1]).emb.data)
            q = QuaternionUtils.normalize_quaternion(q)
            (q_real, q_i, q_j, q_k) = q

            theta = self.create_embedding(self.dim, emb_type=etype, register=False, name="theta",
                                          init_method='uniform', init_params=[-math.pi, math.pi]).emb / 2
            phi = self.create_embedding(self.dim, emb_type=etype, register=False, name="phi").emb
            theta_sin = torch.sin(theta)

            q_real = phi * torch.cos(theta)
            q_i *= phi * theta_sin
            q_j *= phi * theta_sin
            q_k *= phi * theta_sin

            return q_real, q_i, q_j, q_k

        # See: https://github.com/Lion-ZS/DualE/blob/main/DualE-master/DualE-master/models/DualE.py#L239
        for (etype, prefix) in [('entity', 'e'), ('relation', 'r')]:
            # w and t are normalized quaternions with real part equal to zero.
            (w_real, w_i, w_j, w_k), (t_real, t_i, t_j, t_k) = init_quat(etype, prefix), init_quat(etype, prefix)

            self.get_embedding(etype, prefix+'_a0').emb.data = w_real
            self.get_embedding(etype, prefix+'_b0').emb.data = w_i
            self.get_embedding(etype, prefix+'_c0').emb.data = w_j
            self.get_embedding(etype, prefix+'_d0').emb.data = w_k
            self.get_embedding(etype, prefix+'_a1').emb.data = (-t_i * w_i - t_j * w_j - t_k * w_k) / 2
            self.get_embedding(etype, prefix+'_b1').emb.data = (t_i * w_real + t_j * w_k - t_k * t_j) / 2
            self.get_embedding(etype, prefix+'_c1').emb.data = (-t_i * w_k + t_j * w_real + t_k * t_i) / 2
            self.get_embedding(etype, prefix+'_d1').emb.data = (t_i * w_j - t_j * w_i + t_k * w_real) / 2

    def _calc(self, h, r, t):
        def get_real_dual(x):
            ((x_a0, x_a1), (x_b0, x_b1), (x_c0, x_c1), (x_d0, x_d1)) = x
            return (x_a0, x_b0, x_c0, x_d0), (x_a1, x_b1, x_c1, x_d1)

        # Use Schmidt orthogonalization to normalize r. It is unclear in the paper, so we take it from the original
        #   implementation: https://github.com/Lion-ZS/DualE/blob/main/DualE-master/DualE-master/models/DualE.py#L100
        def norm():
            r_real, r_dual = get_real_dual(r)

            # r_real is normalized (see Eq. (3)).
            (a0, b0, c0, d0) = QuaternionUtils.normalize_quaternion(r_real)

            if self.variant == 'full':
                # See https://en.wikipedia.org/wiki/Gram%E2%80%93Schmidt_process#The_Gram%E2%80%93Schmidt_process
                def project(u, v):
                    return (torch.sum(v * u, -1)/torch.sum(u * u, -1)).view(-1, 1) * u

                # Eq. (2) is applying Gram–Schmidt to obtain d dash. It is applied to every component independently.
                d_dash = ()
                for i in range(4):
                    c, d = r_real[i], r_dual[i]
                    d_dash += (d - project(c, d),)
                (a1, b1, c1, d1) = d_dash
            if self.variant == 'cross':
                (a1, b1, c1, d1) = QuaternionUtils.normalize_quaternion(r_dual)

            return (a0, a1), (b0, b1), (c0, c1), (d0, d1)

        h_real, h_dual = get_real_dual(h)
        r_real, r_dual = get_real_dual(norm())
        t_real, t_dual = get_real_dual(t)

        # Eq. (6) implies that Q_hxW_r is the Hamilton product of the dual numbers.
        # Check this: https://faculty.sites.iastate.edu/jia/files/inline-files/dual-quaternion.pdf
        # Let d1=a1+eb1 and d2=a2+eb2 be two dual numbers. Dual number multiplication is d1xd2=a1a2 + e(a1b2 + a2b1).
        hr_prod_real = QuaternionUtils.hamilton_product(h_real, r_real)
        hr_prod_dual = QuaternionUtils.addition(QuaternionUtils.hamilton_product(h_real, r_dual),
                                                QuaternionUtils.hamilton_product(h_dual, r_real))

        # Eq. (8). It seems that these are inner products of real and dual parts:
        #   https://github.com/Lion-ZS/DualE/blob/main/DualE-master/DualE-master/models/DualE.py#L132
        return QuaternionUtils.inner_product(hr_prod_real, t_real) + QuaternionUtils.inner_product(hr_prod_dual, t_dual)

    def return_score(self, is_predict=False):
        (head_emb, rel_emb, tail_emb) = self.current_batch

        h = ((head_emb["e_a0"], head_emb["e_a1"]), (head_emb["e_b0"], head_emb["e_b1"]),
             (head_emb["e_c0"], head_emb["e_c1"]), (head_emb["e_d0"], head_emb["e_d1"]))
        t = ((tail_emb["e_a0"], tail_emb["e_a1"]), (tail_emb["e_b0"], tail_emb["e_b1"]),
             (tail_emb["e_c0"], tail_emb["e_c1"]), (tail_emb["e_d0"], tail_emb["e_d1"]))
        r = ((rel_emb["r_a0"], rel_emb["r_a1"]), (rel_emb["r_b0"], rel_emb["r_b1"]),
             (rel_emb["r_c0"], rel_emb["r_c1"]), (rel_emb["r_d0"], rel_emb["r_d1"]))

        return self._calc(h, r, t)
