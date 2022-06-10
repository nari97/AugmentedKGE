import torch
from Models.Model import Model


class TransERS(Model):
    def __init__(self, ent_total, rel_total, dim, norm=2):
        super(TransERS, self).__init__(ent_total, rel_total)
        self.dim = dim
        self.pnorm = norm

    def get_default_loss(self):
        return 'margin_limit'

    def initialize_model(self):
        self.create_embedding(self.dim, emb_type="entity", name="e", norm_method="norm")
        self.create_embedding(self.dim, emb_type="relation", name="r")

    def _calc(self, h, r, t):
        return -torch.linalg.norm(h + r - t, dim=-1, ord=self.pnorm)

    def return_score(self, is_predict=False):
        (head_emb, rel_emb, tail_emb) = self.current_batch

        h = head_emb["e"]
        t = tail_emb["e"]
        r = rel_emb["r"]

        return self._calc(h, r, t)
