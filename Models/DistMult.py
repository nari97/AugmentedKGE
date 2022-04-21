import torch
from .Model import Model


class DistMult(Model):

    def __init__(self, ent_total, rel_total, dims):
        super(DistMult, self).__init__(ent_total, rel_total, dims, "distmult")

        self.create_embedding(self.dims, emb_type="entity", name="e", norm_method="norm")
        self.create_embedding(self.dims, emb_type="relation", name="r")

        self.register_scale_constraint(emb_type="relation", name="r", p=2)
        
    def _calc(self, h, r, t):
        return torch.sum(h * r * t, -1)

    def return_score(self, head_emb, rel_emb, tail_emb, is_predict=False):
        h = head_emb["e"]
        t = tail_emb["e"]
        r = rel_emb["r"]

        return self._calc(h, r, t)
