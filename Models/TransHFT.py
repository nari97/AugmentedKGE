import torch
from Models.Model import Model


class TransHFT(Model):
    """
    Jun Feng, Minlie Huang, Mingdong Wang, Mantong Zhou, Yu Hao, Xiaoyan Zhu: Knowledge Graph Embedding by Flexible
        Translation. KR 2016: 557-560.
    """
    def __init__(self, ent_total, rel_total, dim):
        """
            dim (int): Number of dimensions for embeddings
        """
        super(TransHFT, self).__init__(ent_total, rel_total)
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
        self.create_embedding(self.dim, emb_type="entity", name="e")
        self.create_embedding(self.dim, emb_type="relation", name="r")
        self.create_embedding(self.dim, emb_type="relation", name="w_r", norm_method="norm")
        self.register_scale_constraint(emb_type="entity", name="e")
        self.register_custom_constraint(TransHFT.orthogonal_constraint)

    @staticmethod
    def orthogonal_constraint(head_emb, rel_emb, tail_emb, eps=1e-5):
        r, w_r = rel_emb["r"], rel_emb["w_r"]
        mult = torch.sum(w_r * r, dim=-1).view(-1, 1)
        constraint = torch.sum(
            torch.pow(mult, 2)/torch.pow(torch.linalg.norm(r, dim=-1, ord=2).view(-1, 1), 2) - eps**2, dim=-1)
        return torch.maximum(constraint, torch.zeros_like(constraint))

    def _calc(self, h, r, t, w_r):
        def transfer(e):
            mult = torch.sum(w_r*e, dim=-1).view(-1, 1)
            return e - mult * w_r
        ht, tt = transfer(h), transfer(t)
        return torch.sum((ht + r) * tt + ht * (tt - r), -1)

    def return_score(self, is_predict=False):
        (head_emb, rel_emb, tail_emb) = self.current_batch

        h = head_emb["e"]
        t = tail_emb["e"]
        r, w_r = rel_emb["r"], rel_emb["w_r"]

        return self._calc(h, r, t, w_r)
