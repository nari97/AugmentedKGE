import torch
from .Model import Model
from Utils.Embedding import Embedding

class DistMult(Model):

    def __init__(self, ent_total, rel_total, dims, norm = 2):
        super(DistMult, self).__init__(ent_total, rel_total)

        self.dims = dims
        self.norm = norm

        self.entities = Embedding(self.ent_tot, self.dims)
        self.relations = Embedding(self.rel_tot, self.dims)

    def normalize(self):
        self.entities.normalize()
        self.relations.normalize()
        

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

        score = self._calc(h,r,t).flatten()

        return score

    def predict(self, data):
        score = -self.forward(data)

        return score



