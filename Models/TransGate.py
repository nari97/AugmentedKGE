import torch
from Models.Model import Model


class TransGate(Model):
    """
    Jun Yuan, Neng Gao, Ji Xiang: TransGate: Knowledge Graph Embedding with Shared Gate Structure. AAAI 2019: 3100-3107.
    """
    def __init__(self, ent_total, rel_total, dim, norm=2, variant='wv'):
        """
            dim (int): Number of dimensions for embeddings
            norm (int): L1 or L2 norm. Default: 2.
            variant can be either fc (fully connected layer) or wv (weight vectors).
        """
        super(TransGate, self).__init__(ent_total, rel_total)
        self.dim = dim
        self.pnorm = norm
        self.variant = variant

        if self.variant == 'fc':
            # Fully connected layers
            self.fch = torch.nn.Linear(dim*2, dim, dtype=torch.float64)
            self.fct = torch.nn.Linear(dim*2, dim, dtype=torch.float64)

        if self.variant == 'wv':
            self.fch = None
            self.fct = None

    def get_default_loss(self):
        # Eq. (8).
        return 'margin'

    def get_score_sign(self):
        # It is a distance (norm).
        return -1

    def initialize_model(self):
        # See above Eq. (8) for norm constraint.
        self.create_embedding(self.dim, emb_type="entity", name="e", norm_method="norm")
        # "We initialize the biases as vectors that all elements are 1."
        self.create_embedding(self.dim, emb_type="entity", name="bh", init_method="uniform", init_params=[1, 1])
        self.create_embedding(self.dim, emb_type="entity", name="bt", init_method="uniform", init_params=[1, 1])
        # See above Eq. (8) for norm constraint.
        self.create_embedding(self.dim, emb_type="relation", name="r", norm_method="norm")

        if self.variant == 'wv':
            # It is unclear whether there is a single v and b for each entity or two: one depending on whether the
            #   entity is head or tail.
            self.create_embedding(self.dim, emb_type="entity", name="vh")
            self.create_embedding(self.dim, emb_type="entity", name="vt")
            self.create_embedding(self.dim, emb_type="relation", name="vrh")
            self.create_embedding(self.dim, emb_type="relation", name="vrt")

    def gate(self, e, v, r, vr, b, layer):
        if self.variant == 'fc':
            # Eqs. (3) and (4). Stack inputs.
            return e * torch.sigmoid(layer(torch.cat([e, r], 1)) + b)
        if self.variant == 'wv':
            # Eqs. (5) and (6).
            return e * torch.sigmoid(v*e + vr*r + b)

    def _calc(self, h, vh, bh, r, vrh, vrt, t, vt, bt, is_predict):
        # These are h_r and t_r in Eq. (7).
        hgate = self.gate(h, vh, r, vrh, bh, self.fch)
        tgate = self.gate(t, vt, r, vrt, bt, self.fct)

        if not is_predict:
            # The paper says ||gate||_2=1; we implement ||gate||_2<=1 and ||gate||_2>=1.
            self.onthefly_constraints.append(self.scale_constraint(hgate))
            self.onthefly_constraints.append(self.scale_constraint(hgate, ctype='ge'))
            self.onthefly_constraints.append(self.scale_constraint(tgate))
            self.onthefly_constraints.append(self.scale_constraint(tgate, ctype='ge'))

        return torch.linalg.norm(hgate + r - tgate, dim=-1, ord=self.pnorm)

    def return_score(self, is_predict=False):
        (head_emb, rel_emb, tail_emb) = self.current_batch

        h, vh, bh = head_emb["e"], head_emb.get("vh", None), head_emb["bh"]
        t, vt, bt = tail_emb["e"], tail_emb.get("vt", None), tail_emb["bt"]
        r, vrh, vrt = rel_emb["r"], rel_emb.get("vrh", None), rel_emb.get("vrt", None)

        return self._calc(h, vh, bh, r, vrh, vrt, t, vt, bt, is_predict)
