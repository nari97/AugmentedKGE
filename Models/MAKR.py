import torch
from Models.Model import Model


class MAKR(Model):
    """
    Yongming Han, GuoFei Chen, Zhongkun Li, Zhiqiang Geng, Fang Li, Bo Ma: An asymmetric knowledge representation
        learning in manifold space. Inf. Sci. 531: 1-12 (2020).
    """
    def __init__(self, ent_total, rel_total, dim, norm=2):
        """
            dim (int): Number of dimensions for embeddings
            norm (int): L1 or L2 norm. Default: 2
        """
        super(MAKR, self).__init__(ent_total, rel_total)
        self.dim = dim
        self.pnorm = norm

    def get_default_loss(self):
        # Eq. (24).
        return 'margin'

    def get_score_sign(self):
        # It is a distance (norm).
        return -1

    def initialize_model(self):
        self.create_embedding(self.dim, emb_type="entity", name="e")
        self.create_embedding(self.dim, emb_type="relation", name="rh")
        self.create_embedding(self.dim, emb_type="relation", name="rt")
        # It is unclear how many relational hyperspheres we have. We assume one per relation.
        self.create_embedding(1, emb_type="relation", name="dr")

        # After Eq. (25).
        self.register_scale_constraint(emb_type="entity", name="e")
        self.register_scale_constraint(emb_type="relation", name="rh")
        self.register_scale_constraint(emb_type="relation", name="rt")

    def _calc(self, h, rh, t, rt, dr):
        # Eqs. (22) and (23).
        return torch.linalg.norm(rh * h - rt * t - torch.pow(dr, 2), dim=-1, ord=self.pnorm)

    def return_score(self, is_predict=False):
        (head_emb, rel_emb, tail_emb) = self.current_batch

        h = head_emb["e"]
        t = tail_emb["e"]
        rh, rt, dr = rel_emb["rh"], rel_emb["rt"], rel_emb["dr"]

        return self._calc(h, rh, t, rt, dr)
