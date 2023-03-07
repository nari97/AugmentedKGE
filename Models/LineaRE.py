import torch
from Models.Model import Model


class LineaRE(Model):
    """
    Yanhui Peng, Jing Zhang: LineaRE: Simple but Powerful Knowledge Graph Embedding for Link Prediction. ICDM 2020:
        422-431.
    """
    def __init__(self, ent_total, rel_total, dim, norm=1):
        """
            dim (int): Number of dimensions for embeddings
            norm (int): L1 or L2 norm. Default: 1 (see Table 1).
        """
        super(LineaRE, self).__init__(ent_total, rel_total)
        self.dim = dim
        self.pnorm = norm

    def get_default_loss(self):
        # Eq. (5).
        return 'soft_margin'

    def get_score_sign(self):
        # It is a distance (norm).
        return -1

    def initialize_model(self):
        # See Table 1. See Eq. (5) for regularization.
        self.create_embedding(self.dim, emb_type="entity", name="e", reg=True)
        self.create_embedding(self.dim, emb_type="relation", name="w_1")
        self.create_embedding(self.dim, emb_type="relation", name="w_2")
        self.create_embedding(self.dim, emb_type="relation", name="b")

    def _calc(self, h, w_1, w_2, b, t):
        # Eq. (2).
        return torch.linalg.norm(w_1 * h + b - w_2 * t, dim=-1, ord=self.pnorm)

    def return_score(self, is_predict=False):
        (head_emb, rel_emb, tail_emb) = self.current_batch

        h = head_emb["e"]
        t = tail_emb["e"]
        w_1, w_2, b = rel_emb["w_1"], rel_emb["w_2"], rel_emb["b"]

        return self._calc(h, w_1, w_2, b, t)
