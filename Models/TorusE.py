import math
import torch
from Models.Model import Model


class TorusE(Model):
    def __init__(self, ent_total, rel_total, dim, norm=2):
        super(TorusE, self).__init__(ent_total, rel_total)
        self.dim = dim
        self.pnorm = norm

    def get_default_loss(self):
        return 'margin'

    def initialize_model(self):
        self.create_embedding(self.dim, emb_type="entity", name="e")
        self.create_embedding(self.dim, emb_type="relation", name="r")

    def _calc(self, h, r, t):
        # From here: https://github.com/TakumaE/TorusE/blob/master/models.py
        d = h + r - t
        d = d - torch.floor(d)
        d = torch.minimum(d, 1 - d)

        if self.pnorm is 1 or 2:
            scores = - 2 * self.pnorm * torch.linalg.norm(d, dim=-1, ord=self.pnorm)
        else:
            scores = 2 - 2 * torch.cos(2 * math.pi * d)/4

        return scores

    def return_score(self, is_predict=False):
        (head_emb, rel_emb, tail_emb) = self.current_batch

        h = head_emb["e"]
        t = tail_emb["e"]
        r = rel_emb["r"]

        return self._calc(h, r, t)
