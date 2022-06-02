import torch
from Models.Model import Model


class TransGate(Model):
    def __init__(self, ent_total, rel_total, dim, norm=2):
        super(TransGate, self).__init__(ent_total, rel_total)
        self.dim = dim
        self.pnorm = norm

    def get_default_loss(self):
        return 'margin'

    def initialize_model(self):
        self.create_embedding(self.dim, emb_type="entity", name="e", norm_method="norm")
        self.create_embedding(self.dim, emb_type="entity", name="v")
        self.create_embedding(self.dim, emb_type="entity", name="b")
        self.create_embedding(self.dim, emb_type="relation", name="r", norm_method="norm")
        self.create_embedding(self.dim, emb_type="relation", name="vrh")
        self.create_embedding(self.dim, emb_type="relation", name="vrt")

        # Even though the paper says ||gate||_2=1, we implement ||gate||_2<=1.
        self.register_custom_constraint(self.h_constraint)
        self.register_custom_constraint(self.t_constraint)

    def h_constraint(self, head_emb, rel_emb, tail_emb):
        h, vh, bh = head_emb["e"], head_emb["v"], head_emb["b"]
        r, vrh = rel_emb["r"], rel_emb["vrh"]
        return self.max_clamp(torch.linalg.norm(self.gate(h, vh, r, vrh, bh), dim=-1, ord=2), 1)

    def t_constraint(self, head_emb, rel_emb, tail_emb):
        t, vt, bt = tail_emb["e"], tail_emb["v"], tail_emb["b"]
        r, vrt = rel_emb["r"], rel_emb["vrt"]
        return self.max_clamp(torch.linalg.norm(self.gate(t, vt, r, vrt, bt), dim=-1, ord=2), 1)

    def gate(self, e, v, r, vr, b):
        return e * torch.sigmoid(v*e + vr*r + b)

    def _calc(self, h, vh, bh, r, vrh, vrt, t, vt, bt):
        return -torch.linalg.norm(self.gate(h, vh, r, vrh, bh) + r - self.gate(t, vt, r, vrt, bt), dim=-1, ord=self.pnorm)

    def return_score(self, is_predict=False):
        (head_emb, rel_emb, tail_emb) = self.current_batch

        h, vh, bh = head_emb["e"], head_emb["v"], head_emb["b"]
        t, vt, bt = tail_emb["e"], tail_emb["v"], tail_emb["b"]
        r, vrh, vrt = rel_emb["r"], rel_emb["vrh"], rel_emb["vrt"]

        return self._calc(h, vh, bh, r, vrh, vrt, t, vt, bt)
