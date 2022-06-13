import torch
from .Model import Model


class SimplE(Model):

    def __init__(self, ent_total, rel_total, dim):
        super(SimplE, self).__init__(ent_total, rel_total)
        self.dim = dim

    def get_default_loss(self):
        return 'soft'

    def initialize_model(self):
        self.create_embedding(self.dim, emb_type="entity", name="he", reg=True)
        self.create_embedding(self.dim, emb_type="entity", name="te", reg=True)
        self.create_embedding(self.dim, emb_type="relation", name="r", reg=True)
        self.create_embedding(self.dim, emb_type="relation", name="r_inv", reg=True)

    def _calc_avg(self, hei, hej, tei, tej, r, r_inv):
        return (torch.sum(hei * r * tej, -1) + torch.sum(hej * r_inv * tei, -1))/2

    def _calc_ingr(self, h, r, t):
        return torch.sum(h * r * t, -1)

    def return_score(self, is_predict=False):
        (head_emb, rel_emb, tail_emb) = self.current_batch

        hei, hej = head_emb["he"], tail_emb["he"]
        tei, tej = head_emb["te"], tail_emb["te"]
        r, r_inv = rel_emb["r"], rel_emb["r_inv"]

        if is_predict:
            return self._calc_ingr(hei, r, tej)
        else:
            return self._calc_avg(hei, hej, tei, tej, r, r_inv)



