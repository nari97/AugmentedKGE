import numpy as np
import torch
from Models.Model import Model
from Utils import PoincareUtils


# https://github.com/tensorflow/neural-structured-learning/blob/master/research/kg_hyp_emb/models/hyperbolic.py#L116
class AttH(Model):
    """
    Ines Chami, Adva Wolf, Da-Cheng Juan, Frederic Sala, Sujith Ravi, Christopher Ré: Low-Dimensional Hyperbolic
        Knowledge Graph Embeddings. ACL 2020: 6901-6914.
    """
    def __init__(self, ent_total, rel_total, dim, variant="atth"):
        """
            dim (int): Number of dimensions for embeddings
            variant can be either atth, roth, refh, atte, rote, refe; *h means using Poincare, *e means using Euclidean,
                att* means using the full proposal, and rot* and ref* mean using rotation and reflection only, resp.
        """
        super(AttH, self).__init__(ent_total, rel_total)
        # It must be divided by two because of the theta and phi embeddings.
        self.dim = 2*int(dim//2)
        self.variant = variant

    def get_default_loss(self):
        # Eq. (11).
        return 'soft'

    def get_score_sign(self):
        # It is a distance.
        return -1

    def initialize_model(self):
        self.create_embedding(self.dim, emb_type="entity", name="e")
        self.create_embedding(1, emb_type="entity", name="b")

        self.create_embedding(self.dim, emb_type="relation", name="r")
        # Only when using full or rotation.
        if self.variant.startswith('att') or self.variant.startswith('rot'):
            self.create_embedding(int(self.dim/2), emb_type="relation", name="theta")
        # Only when using full or reflection.
        if self.variant.startswith('att') or self.variant.startswith('ref'):
            self.create_embedding(int(self.dim/2), emb_type="relation", name="phi")
        self.create_embedding(self.dim, emb_type="relation", name="a")
        # Only when using Poincare.
        if self.variant.endswith('h'):
            self.create_embedding(1, emb_type="relation", name="c")

    def rotation_multiplication(self, r, e):
        # r = [r0, r1]
        # r as a block-diagonal matrix:
        #  cos r0, -sin r0,      0,       0
        #  sin r0,  cos r0,      0,       0
        #       0,       0, cos r1, -sin r1
        #       0,       0, sin r1,  cos r1
        # e = [e0, e1, e2, e3]
        # r*e = [e0 cos r0 + e1 -sin r0, e1 cos r0 + e0 sin r0, e2 cos r1 + e3 -sin r1, e3 cos r1 + e2 sin r1]
        # r_diag = [cos r0, cos r0, cos r1, cos r1] (of the block-diagonal matrix)
        # r_diag*e gives all the first elements in r*e: [e0 cos r0, e1 cos r0, e2 cos r2, e3 cos r2] => diag_times_e
        # r_cdiag = [sin r0, sin r0, sin r1, sin r1] (of the block-diagonal matrix)
        # r_cdiag*e gives all the second elements in r*h with no sign and in different order:
        #   [e0 sin r0, e1 sin r0, e2 sin r1, e3 sin r1] => cdiag_times_e
        # diag_times_e[even] -= cdiag_times_e[odd] (the even positions of the result)
        # diag_times_e[odd] += cdiag_times_e[even] (the odd positions of the result)

        # Even and odd indexes.
        even_indexes = torch.LongTensor(np.arange(0, self.dim, 2))
        odd_indexes = torch.LongTensor(np.arange(1, self.dim, 2))

        r_diag = torch.cos(r).repeat_interleave(2, dim=1)
        diag_times_e = r_diag*e

        r_cdiag = torch.sin(r).repeat_interleave(2, dim=1)
        cdiag_times_e = r_cdiag*e

        diag_times_e[:, even_indexes] -= cdiag_times_e[:, odd_indexes]
        diag_times_e[:, odd_indexes] += cdiag_times_e[:, even_indexes]

        return diag_times_e

    def reflection_multiplication(self, r, e):
        # r = [r0, r1]
        # r as a block-diagonal matrix:
        #  cos r0,  sin r0,      0,       0
        #  sin r0, -cos r0,      0,       0
        #       0,       0, cos r1,  sin r1
        #       0,       0, sin r1, -cos r1
        # e = [e0, e1, e2, e3]
        # r*e = [e0 cos r0 + e1 sin r0, e1 -cos r0 + e0 sin r0, e2 cos r1 + e3 sin r1, e3 -cos r1 + e2 sin r1]
        # r_diag = [cos r0, cos r0, cos r2, cos r2] (of the block-diagonal matrix)
        # r_diag[even] *= -1
        # r_diag*e gives all the first elements in r*e: [e0 cos r0, e1 -cos r0, e2 cos r1, e3 -cos r1] => diag_times_e
        # r_cdiag = [sin r0, sin r0, sin r1, sin r1] (of the block-diagonal matrix)
        # r_cdiag*e gives all the second elements in r*h in different order:
        #   [e0 sin r0, e1 sin r0, e2 sin r1, e3 sin r1] => cdiag_times_e
        # diag_times_e[even] += cdiag_times_e[odd] (the even positions of the result)
        # diag_times_e[odd] += cdiag_times_e[even] (the odd positions of the result)

        # Even and odd indexes.
        even_indexes = torch.LongTensor(np.arange(0, self.dim, 2))
        odd_indexes = torch.LongTensor(np.arange(1, self.dim, 2))

        r_diag = torch.cos(r).repeat_interleave(2, dim=1)
        r_diag[:, even_indexes] *= -1
        diag_times_e = r_diag*e

        r_cdiag = torch.sin(r).repeat_interleave(2, dim=1)
        cdiag_times_e = r_cdiag*e

        diag_times_e[:, even_indexes] += cdiag_times_e[:, odd_indexes]
        diag_times_e[:, odd_indexes] += cdiag_times_e[:, even_indexes]

        return diag_times_e

    def get_attention(self, e_h, c, a):
        if self.variant.endswith('h'):
            # See above Eq. (7). Map to Tangent and compute a^T*e_tangent
            e_tangent = PoincareUtils.log_map(e_h, c)
        else:
            e_tangent = e_h
        return e_tangent, torch.sum(a * e_tangent, -1)

    # To train model in the hyperbolic, we need a special SGD (see
    #   https://github.com/ibalazevic/multirelational-poincare/blob/master/rsgd.py).
    # Instead, we optimize in the tangent space and map them to the Poincare ball. Check Section A.4 in the paper.
    def _calc(self, h, bh, r, theta, phi, a, c, t, bt):
        if self.variant.endswith('h'):
            # c must be positive and is only present in dealing with Poincare.
            c = torch.abs(c)
            # Map h, r and t from Tangent to Poincare.
            h, r, t = PoincareUtils.exp_map(h, c), PoincareUtils.exp_map(r, c), PoincareUtils.exp_map(t, c)

        # Rotate and reflect. Eqs. (4), (5), (6) and (8).
        if self.variant.startswith('att') or self.variant.startswith('rot'):
            h_rot = self.rotation_multiplication(theta, h)
        else:
            h_rot = h
        if self.variant.startswith('att') or self.variant.startswith('ref'):
            h_ref = self.reflection_multiplication(phi, h)
        else:
            h_ref = h

        # Compute alpha.
        (tangent_rot, alpha_rot), (tangent_ref, alpha_ref) = self.get_attention(h_rot, c, a), \
            self.get_attention(h_ref, c, a)
        # Alpha has two positions.
        alpha = torch.nn.functional.softmax(torch.cat((alpha_rot.view(-1, 1), alpha_ref.view(-1, 1)), dim=1), dim=-1)
        # Attention. Eq. (7).
        att = alpha[:, 0].view(-1, 1) * tangent_rot + alpha[:, 1].view(-1, 1) * tangent_ref
        if self.variant.endswith('h'):
            att = PoincareUtils.exp_map(att, c)
        if self.variant.endswith('h'):
            # Eq. (9).
            hr = PoincareUtils.mobius_addition(att, r, c)
        else:
            # Just regular addition.
            hr = att + r

        if self.variant.endswith('h'):
            # Eq. (10) with different sign.
            scores = PoincareUtils.geodesic_dist(hr, t, c)**2 - bh - bt
        else:
            # Just simple Euclidean norm.
            scores = torch.linalg.norm(-hr + t, dim=-1, ord=2)**2 - bh.flatten() - bt.flatten()
        return scores

    def return_score(self, is_predict=False):
        (head_emb, rel_emb, tail_emb) = self.current_batch

        h, bh = head_emb["e"], head_emb["b"]
        t, bt = tail_emb["e"], tail_emb["b"]
        r, theta, phi, a, c = rel_emb["r"], rel_emb.get("theta", None), rel_emb.get("phi", None), rel_emb["a"], \
            rel_emb.get("c", None)

        return self._calc(h, bh, r, theta, phi, a, c, t, bt)
