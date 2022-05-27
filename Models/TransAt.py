import torch
from Models.Model import Model


class TransAt(Model):
    def __init__(self, ent_total, rel_total, dim, norm=2):
        super(TransAt, self).__init__(ent_total, rel_total)
        self.dim = dim
        self.pnorm = norm

    def get_default_loss(self):
        return 'margin'

    def initialize_model(self):
        self.create_embedding(self.dim, emb_type="entity", name="e")
        self.create_embedding(self.dim, emb_type="relation", name="r")
        self.create_embedding(self.dim, emb_type="relation", name="rh")
        self.create_embedding(self.dim, emb_type="relation", name="rt")

        self.register_scale_constraint(emb_type="entity", name="e", p=2)
        self.register_scale_constraint(emb_type="relation", name="r", p=2)

    def proj(self, a, x):
        return a*x

    def _calc(self, h, r, rh, rt, t):
        # The orignal paper does not use norm but, then, the output are not scores.
        return -torch.linalg.norm(self.proj(torch.sigmoid(rh), h) + r - self.proj(torch.sigmoid(rt), t),
                                  dim=-1, ord=self.pnorm)

    def return_score(self, is_predict=False):
        (head_emb, rel_emb, tail_emb) = self.current_batch

        h = head_emb["e"]
        t = tail_emb["e"]
        r = rel_emb["r"]
        rh, rt = rel_emb["rh"], rel_emb["rt"]

        return self._calc(h, r, rh, rt, t)
