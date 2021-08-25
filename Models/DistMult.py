import torch
from .Model import Model
from Utils.Embedding import Embedding
import torch.nn.functional as F

class DistMult(Model):

    def __init__(self, ent_total, rel_total, dims, norm = 2, inner_norm = False):
        super(DistMult, self).__init__(ent_total, rel_total)

        self.dims = dims
        self.norm = norm
        self.inner_norm = inner_norm

        self.entities = Embedding(self.ent_tot, self.dims)
        self.relations = Embedding(self.rel_tot, self.dims)

    def normalize(self):
        self.entities.normalize()
        self.relations.normalize()
        
    def normalize_inner(self, h, r, t):
        h = F.normalize(h, dim = -1, p = 2)
        r = F.normalize(r, dim = -1, p = 2)
        t = F.normalize(t, dim = -1, p = 2)

        return h,r,t

    def _calc(self, h,r,t):
        score = (h * r) * t
        score = torch.sum(score, -1)
        return score

    def forward(self, data):

        batch_h = self.get_batch(data, "h")
        batch_r = self.get_batch(data, "r")
        batch_t = self.get_batch(data, "t")

        h = self.entities.get_embedding(batch_h)
        r = self.relations.get_embedding(batch_r)
        t = self.entities.get_embedding(batch_t)

        if self.inner_norm:
            h,r,t = self.normalize_inner(h,r,t)

        score = self._calc(h,r,t).flatten()

        return score

    def predict(self, data):
        score = -self.forward(data)

        return score



