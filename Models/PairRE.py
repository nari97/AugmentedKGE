import torch
from Models.Model import Model


class PairRE(Model):
    """
    Linlin Chao, Jianshan He, Taifeng Wang, Wei Chu: PairRE: Knowledge Graph Embeddings via Paired Relation Vectors.
        ACL/IJCNLP (1) 2021: 4360-4369.
    """
    def __init__(self, ent_total, rel_total, dim, norm=1):
        """
            dim (int): Number of dimensions for embeddings
            norm (int): L1 or L2 norm. Default: 1 ("In this paper, we take the L1-norm to measure the distance.")
        """
        super(PairRE, self).__init__(ent_total, rel_total)
        self.dim = dim
        self.pnorm = norm

    def get_default_loss(self):
        # Eq. (8).
        return 'soft_margin'

    def get_score_sign(self):
        # It is a distance (norm).
        return -1

    def initialize_model(self):
        # See below Eq. (1).
        self.create_embedding(self.dim, emb_type="entity", name="e", norm_method="norm")
        self.create_embedding(self.dim, emb_type="relation", name="rh")
        self.create_embedding(self.dim, emb_type="relation", name="rt")

    def _calc(self, h, rh, rt, t):
        # Eq. (1).
        return torch.linalg.norm(h * rh - t * rt, dim=-1, ord=self.pnorm)

    def return_score(self, is_predict=False):
        (head_emb, rel_emb, tail_emb) = self.current_batch

        h = head_emb["e"]
        t = tail_emb["e"]
        rh, rt = rel_emb["rh"], rel_emb["rt"]

        return self._calc(h, rh, rt, t)
