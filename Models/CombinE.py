import math
import torch
from Models.Model import Model


class CombinE(Model):
    """
    Zhen Tan, Xiang Zhao, Wei Wang: Representation Learning of Large-Scale Knowledge Graphs via Entity Feature
        Combinations. CIKM 2017: 1777-1786.
    """
    def __init__(self, ent_total, rel_total, dim, norm=2):
        """
            dim (int): Number of dimensions for embeddings
            norm (int): L1 or L2 norm. Default: 2
        """
        super(CombinE, self).__init__(ent_total, rel_total)
        self.dim = dim
        self.pnorm = norm

    def get_default_loss(self):
        # Eq. (5)
        return 'margin'

    def get_score_sign(self):
        # It is a norm.
        return -1

    def initialize_model(self):
        # Section 3.
        # Eq. (8) proposes L1/L2 regularization.
        self.create_embedding(self.dim, emb_type="entity", name="ep", reg=True)
        self.create_embedding(self.dim, emb_type="entity", name="em", reg=True)
        self.create_embedding(self.dim, emb_type="relation", name="rp", reg=True)
        self.create_embedding(self.dim, emb_type="relation", name="rm", reg=True)

        # Eq. (6).
        self.register_scale_constraint(emb_type="entity", name="ep")
        self.register_scale_constraint(emb_type="entity", name="em")
        # Eq. (7).
        self.register_scale_constraint(emb_type="relation", name="rp", z=math.sqrt(2))
        self.register_scale_constraint(emb_type="relation", name="rm")

    def _calc(self, hp, hm, rp, rm, tp, tm):
        # Eq. (3)
        return torch.pow(torch.linalg.norm(hp + tp - rp, dim=-1, ord=self.pnorm), 2) + \
                    torch.pow(torch.linalg.norm(hm - tm - rm, dim=-1, ord=self.pnorm), 2)

    def return_score(self, is_predict=False):
        (head_emb, rel_emb, tail_emb) = self.current_batch

        hp, hm = head_emb["ep"], head_emb["em"]
        tp, tm = tail_emb["ep"], tail_emb["em"]
        rp, rm = rel_emb["rp"], rel_emb["rm"]

        return self._calc(hp, hm, rp, rm, tp, tm)
