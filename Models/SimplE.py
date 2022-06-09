import torch
from .Model import Model


class SimplE(Model):

    def __init__(self, ent_total, rel_total, dim):
        super(SimplE, self).__init__(ent_total, rel_total)
        self.dim = dim

    def get_default_loss(self):
        return 'soft'

    def initialize_model(self):
        self.create_embedding(self.dim, emb_type="entity", name="he")
        self.create_embedding(self.dim, emb_type="entity", name="te")
        self.create_embedding(self.dim, emb_type="relation", name="r")
        self.create_embedding(self.dim, emb_type="relation", name="r_inv")

        self.register_scale_constraint(emb_type="entity", name="he")
        self.register_scale_constraint(emb_type="entity", name="te")
        self.register_scale_constraint(emb_type="relation", name="r")
        self.register_scale_constraint(emb_type="relation", name="r_inv")

    def _calc_avg(self, hei, hej, tei, tej, r, r_inv):
        return (torch.sum(hei * r * tej, -1) + torch.sum(hej * r_inv * tei, -1)).flatten()/2

    def _calc_ingr(self, h, r, t):
        return torch.sum(h * r * t, -1)

    def return_score(self, is_predict=False):
        (head_emb, rel_emb, tail_emb) = self.current_batch

        hei = head_emb["he"]
        hej = tail_emb["he"]
        tei = head_emb["te"]
        tej = tail_emb["te"]

        r = rel_emb["r"]
        r_inv = rel_emb["r_inv"]

        if is_predict:
            return self._calc_ingr(hei, r, tej)
        else:
            return self._calc_avg(hei, hej, tei, tej, r, r_inv)



