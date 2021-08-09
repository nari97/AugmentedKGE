import torch
from .Model import Model
from Utils.Embedding import Embedding

class TransH(Model):

    def __init__(self, ent_total, rel_total, dims, norm = 2):
        super(TransH, self).__init__(ent_total, rel_total)

        self.dims = dims
        self.norm = norm

        self.entities = Embedding(self.ent_tot, self.dims)
        self.relations = Embedding(self.rel_tot, self.dims)
        self.norm_vector = Embedding(self.rel_tot, self.dims)

    def normalize(self):
        self.entities.normalize()
        self.norm_vector.normalize()

    def _calc(self, h, r, t, w_r):
        ht = h - torch.sum(h*w_r, dim = -1, keepdim = True).repeat(1, self.dims)*w_r
        tt = t - torch.sum(t*w_r, dim = -1, keepdim = True).repeat(1, self.dims)*w_r
        answer = -torch.pow(torch.norm(ht + r - tt, dim = -1, p = 2),2)

        return answer

    def forward(self, data):

        batch_h = self.get_batch(data, "h")
        batch_r = self.get_batch(data, "r")
        batch_t = self.get_batch(data, "t")

        

        h = self.entities.get_embedding(batch_h)
        r = self.relations.get_embedding(batch_r)
        t = self.entities.get_embedding(batch_t)
        w_r = self.norm_vector.get_embedding(batch_r)
        
        score = self._calc(h, r, t, w_r).flatten()

        return score

    def predict(self, data):
        score = -self.forward(data)

        return score



