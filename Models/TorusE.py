import math
import torch
from Models.Model import Model


class TorusE(Model):
    def __init__(self, ent_total, rel_total, dims, norm=2):
        super(TorusE, self).__init__(ent_total, rel_total, dims, "toruse")

        self.pnorm = norm
        self.create_embedding(self.dims, emb_type="entity", name="e")
        self.create_embedding(self.dims, emb_type="relation", name="r")

    def _calc(self, h, r, t):
        # From here: https://github.com/TakumaE/TorusE/blob/master/models.py
        d = h + r - t
        d = d - torch.floor(d)
        d = torch.minimum(d, 1 - d)

        if self.pnorm is 1 or 2:
            scores = 2 * self.pnorm * torch.linalg.norm(d, dim=-1, ord=self.pnorm)
        else:
            scores = 2 - 2 * torch.cos(2 * math.pi * d)/4

        return -scores

    def return_score(self, head_emb, rel_emb, tail_emb, is_predict=False):
        h = head_emb["e"]
        t = tail_emb["e"]
        r = rel_emb["r"]

        return self._calc(h, r, t)
