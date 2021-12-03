import torch
from .Model import Model
import torch.nn.functional as F
from Utils.NormUtils import clamp_norm, normalize

class DistMult(Model):

    def __init__(self, ent_total, rel_total, dims, norm = 2, inner_norm = False):
        super(DistMult, self).__init__(ent_total, rel_total, dims, "distmult", inner_norm)

        self.create_embedding(self.dims, emb_type = "entity", name = "e", normMethod = "norm", norm_params = self.norm_params)
        self.create_embedding(self.dims, emb_type = "relation", name = "r", normMethod = "clamp", norm_params= self.norm_params)
        
        self.register_params()
        

    def _calc(self, h,r,t):
        score = (h * r) * t
        score = torch.sum(score, -1)
        return score

    def returnScore(self, head_emb, rel_emb, tail_emb):

        h = head_emb["e"]
        t = tail_emb["e"]
        r = rel_emb["r"]


        score = self._calc(h,r,t).flatten()

        return score



