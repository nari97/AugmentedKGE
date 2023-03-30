import torch
from Models.Model import Model
from Utils import PoincareUtils, GivensUtils


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
            h_rot = GivensUtils.rotation_multiplication(theta, h, self.dim)
        else:
            h_rot = h
        if self.variant.startswith('att') or self.variant.startswith('ref'):
            h_ref = GivensUtils.self.reflection_multiplication(phi, h, self.dim)
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
