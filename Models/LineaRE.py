import torch
from Models.Model import Model


class LineaRE(Model):
    def __init__(self, ent_total, rel_total, dim, norm=1):
        super(LineaRE, self).__init__(ent_total, rel_total)
        self.dim = dim
        self.pnorm = norm

    def get_default_loss(self):
        return 'soft_margin'

    def initialize_model(self):
        self.create_embedding(self.dim, emb_type="entity", name="e", reg=True)
        self.create_embedding(self.dim, emb_type="relation", name="w_1")
        self.create_embedding(self.dim, emb_type="relation", name="w_2")
        self.create_embedding(self.dim, emb_type="relation", name="b")

    def _calc(self, h, w_1, w_2, b, t):
        return -torch.linalg.norm(w_1 * h + b - w_2 * t, dim=-1, ord=self.pnorm)

    def return_score(self, is_predict=False):
        (head_emb, rel_emb, tail_emb) = self.current_batch

        h = head_emb["e"]
        t = tail_emb["e"]
        w_1, w_2, b = rel_emb["w_1"], rel_emb["w_2"], rel_emb["b"]

        return self._calc(h, w_1, w_2, b, t)
