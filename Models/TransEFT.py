import torch
from Models.Model import Model


class TransEFT(Model):
    """
    Jun Feng, Minlie Huang, Mingdong Wang, Mantong Zhou, Yu Hao, Xiaoyan Zhu: Knowledge Graph Embedding by Flexible
        Translation. KR 2016: 557-560.
    """

    def __init__(self, ent_total, rel_total, dim):
        """
            dim (int): Number of dimensions for embeddings
        """
        super(TransEFT, self).__init__(ent_total, rel_total)
        self.dim = dim

    def get_default_loss(self):
        # Eq. (1).
        return 'margin'

    def get_score_sign(self):
        # It is a similarity. From the paper: "...we design a new function to score the compatibility of a triple by the
        #   inner product between the sum of head entity vector and relation vector h + r and tail vector t instead of
        #   using the Manhattan/Euclidean distance..."
        return 1

    def initialize_model(self):
        self.create_embedding(self.dim, emb_type="entity", name="e", norm_method="norm")
        self.create_embedding(self.dim, emb_type="relation", name="r")

    def _calc(self, h, r, t):
        # Eq. (2).
        return torch.sum((h + r) * t + h * (t - r), -1)

    def return_score(self, is_predict=False):
        (head_emb, rel_emb, tail_emb) = self.current_batch

        h = head_emb["e"]
        t = tail_emb["e"]
        r = rel_emb["r"]

        return self._calc(h, r, t)