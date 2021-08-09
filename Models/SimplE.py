import torch
from .Model import Model
from Utils.Embedding import Embedding

class SimplE(Model):

    def __init__(self, ent_total, rel_total, dims):
        super(SimplE, self).__init__(ent_total, rel_total)

        self.dims = dims


        self.entities_h = Embedding(self.ent_tot, self.dims)
        self.entities_t = Embedding(self.ent_tot, self.dims)
        self.relations = Embedding(self.rel_tot, self.dims)
        self.relation_inverse = Embedding(self.rel_tot, self.dims)


    def normalize(self):
        #self.entities.normalize()
        pass
        

    def _calc_avg(self, h_i, t_i, h_j, t_j, r, r_inv):
        return (torch.sum(h_i * r * t_j, -1) + torch.sum(h_j * r_inv * t_i, -1))/2

    def _calc_ingr(self, h, r, t):
        return torch.sum(h * r * t, -1)

    def forward(self, data):

        batch_h = self.get_batch(data, "h")
        batch_r = self.get_batch(data, "r")
        batch_t = self.get_batch(data, "t")

        h_i = self.entities_h.get_embedding(batch_h)
        h_j = self.entities_h.get_embedding(batch_t)

        t_i = self.entities_t.get_embedding(batch_h)
        t_j = self.entities_t.get_embedding(batch_t)

        r = self.relations.get_embedding(batch_r)
        r_inv = self.relation_inverse.get_embedding(batch_r)

        score = self._calc_avg(h_i, t_i,h_j, t_j, r, r_inv).flatten()

        return score

    def predict(self, data):
        batch_h = self.get_batch(data, "h")
        batch_r = self.get_batch(data, "r")
        batch_t = self.get_batch(data, "t")

        h = self.entities_h.get_embedding(batch_h)
        r = self.relations.get_embedding(batch_r)
        t = self.entities_t.get_embedding(batch_t)

        score = -self._calc_ingr(h, r, t)
        return score



