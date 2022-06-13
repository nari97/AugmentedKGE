import torch
from Models.Model import Model


class STransE(Model):

    def __init__(self, ent_total, rel_total, dim, norm=2):
        super(STransE, self).__init__(ent_total, rel_total)
        self.dim = dim
        self.pnorm = norm

    def get_default_loss(self):
        return 'margin'

    def initialize_model(self):
        self.create_embedding(self.dim, emb_type="entity", name="e")
        self.create_embedding(self.dim, emb_type="relation", name="r")
        self.create_embedding((self.dim, self.dim), emb_type="relation", name="wr1")
        self.create_embedding((self.dim, self.dim), emb_type="relation", name="wr2")

        self.register_scale_constraint(emb_type="entity", name="e")
        self.register_scale_constraint(emb_type="relation", name="r")

    def get_et(self, w, e):
        batch_size = e.shape[0]
        #  multiply by vector and put it back to regular shape.
        return torch.matmul(w, e.view(batch_size, -1, 1)).view(batch_size, self.dim)

    def _calc(self, h, wr1, r, t, wr2, is_predict):
        ht = self.get_et(wr1, h)
        tt = self.get_et(wr2, t)

        if not is_predict:
            self.onthefly_constraints.append(self.scale_constraint(ht))
            self.onthefly_constraints.append(self.scale_constraint(tt))

        return -torch.linalg.norm(ht + r - tt, dim=-1, ord=self.pnorm)

    def return_score(self, is_predict=False):
        (head_emb, rel_emb, tail_emb) = self.current_batch

        h = head_emb["e"]
        t = tail_emb["e"]
        r, wr1, wr2 = rel_emb["r"], rel_emb["wr1"], rel_emb["wr2"]

        return self._calc(h, wr1, r, t, wr2, is_predict)
