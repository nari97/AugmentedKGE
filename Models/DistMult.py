import torch
from .Model import Model
from Utils.Embedding import Embedding
import torch.nn.functional as F
from Utils.utils import clamp_norm, normalize

class DistMult(Model):

    def __init__(self, ent_total, rel_total, dims, norm = 2, inner_norm = False):
        super(DistMult, self).__init__(ent_total, rel_total)

        self.dims = dims
        self.norm = norm
        self.inner_norm = inner_norm

        norm_params = {"p" : 2, "dim" : -1, "maxnorm" : 1}

        self.create_embedding(self.ent_tot, self.dims, emb_type = "entity", name = "e", normMethod = "norm", norm_params = norm_params)
        
        self.create_embedding(self.rel_tot, self.dims, emb_type = "relation", name = "r", normMethod = "clamp", norm_params= norm_params)
        
    def normalize_inner(self, h, r, t):
        h = normalize(h, dim = -1, p = 2)
        r = clamp_norm(r, dim = -1, p = 2, maxnorm = 1)
        t = normalize(t, dim = -1, p = 2)

        return h,r,t

    def _calc(self, h,r,t):
        score = (h * r) * t
        score = torch.sum(score, -1)
        return score

    def returnScore(self, head_emb, rel_emb, tail_emb):

        h = head_emb["e"]
        t = tail_emb["e"]
        r = rel_emb["r"]

        if self.inner_norm:
            h,r,t = self.normalize_inner(h,r,t)

        score = self._calc(h,r,t).flatten()

        return score



