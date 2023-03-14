import torch
from Models.Model import Model


class TransGate(Model):
    """
    Jun Yuan, Neng Gao, Ji Xiang: TransGate: Knowledge Graph Embedding with Shared Gate Structure. AAAI 2019: 3100-3107.
    """
    def __init__(self, ent_total, rel_total, dim, norm=2):
        super(TransGate, self).__init__(ent_total, rel_total)
        self.dim = dim
        self.pnorm = norm

    def get_default_loss(self):
        # Eq. (8).
        return 'margin'

    def get_score_sign(self):
        # It is a distance (norm).
        return -1

    def initialize_model(self):
        # See above Eq. (8) for norm constraint.
        self.create_embedding(self.dim, emb_type="entity", name="e", norm_method="norm")
        # It is unclear whether there is a single v and b for each entity or two: one depending on whether the entity is
        #   head or tail.
        self.create_embedding(self.dim, emb_type="entity", name="vh")
        self.create_embedding(self.dim, emb_type="entity", name="vt")
        # "We initialize the biases as vectors that all elements are 1."
        self.create_embedding(self.dim, emb_type="entity", name="bh", init_method="uniform", init_params=[1, 1])
        self.create_embedding(self.dim, emb_type="entity", name="bt", init_method="uniform", init_params=[1, 1])
        # See above Eq. (8) for norm constraint.
        self.create_embedding(self.dim, emb_type="relation", name="r", norm_method="norm")
        self.create_embedding(self.dim, emb_type="relation", name="vrh")
        self.create_embedding(self.dim, emb_type="relation", name="vrt")

    def gate(self, e, v, r, vr, b):
        # Eqs. (5) and (6).
        return e * torch.sigmoid(v*e + vr*r + b)

    def _calc(self, h, vh, bh, r, vrh, vrt, t, vt, bt, is_predict):
        # These are h_r and t_r in Eq. (7).
        hgate = self.gate(h, vh, r, vrh, bh)
        tgate = self.gate(t, vt, r, vrt, bt)

        if not is_predict:
            # The paper says ||gate||_2=1; we implement ||gate||_2<=1 and ||gate||_2>=1.
            self.onthefly_constraints.append(self.scale_constraint(hgate))
            self.onthefly_constraints.append(self.scale_constraint(hgate, ctype='ge'))
            self.onthefly_constraints.append(self.scale_constraint(tgate))
            self.onthefly_constraints.append(self.scale_constraint(tgate, ctype='ge'))

        return torch.linalg.norm(hgate + r - tgate, dim=-1, ord=self.pnorm)

    def return_score(self, is_predict=False):
        (head_emb, rel_emb, tail_emb) = self.current_batch

        h, vh, bh = head_emb["e"], head_emb["vh"], head_emb["bh"]
        t, vt, bt = tail_emb["e"], tail_emb["vt"], tail_emb["bt"]
        r, vrh, vrt = rel_emb["r"], rel_emb["vrh"], rel_emb["vrt"]

        return self._calc(h, vh, bh, r, vrh, vrt, t, vt, bt, is_predict)
