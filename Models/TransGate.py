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

    def gate(self, e, v, r, vr, b):
        return e * torch.sigmoid(v*e + vr*r + b)

    def _calc(self, h, vh, bh, r, vrh, vrt, t, vt, bt, is_predict):
        hgate = self.gate(h, vh, r, vrh, bh)
        tgate = self.gate(t, vt, r, vrt, bt)

        if not is_predict:
            # The paper says ||gate||_2=1; we implement ||gate||_2<=1 and ||gate||_2>=1.
            self.onthefly_constraints.append(self.scale_constraint(hgate))
            self.onthefly_constraints.append(self.scale_constraint(hgate, ctype='ge'))
            self.onthefly_constraints.append(self.scale_constraint(tgate))
            self.onthefly_constraints.append(self.scale_constraint(tgate, ctype='ge'))

        return -torch.linalg.norm(hgate + r - tgate, dim=-1, ord=self.pnorm)

    def return_score(self, is_predict=False):
        (head_emb, rel_emb, tail_emb) = self.current_batch

        h, vh, bh = head_emb["e"], head_emb["v"], head_emb["b"]
        t, vt, bt = tail_emb["e"], tail_emb["v"], tail_emb["b"]
        r, vrh, vrt = rel_emb["r"], rel_emb["vrh"], rel_emb["vrt"]

        return self._calc(h, vh, bh, r, vrh, vrt, t, vt, bt, is_predict)
