import math
import torch
from Models.Model import Model


class TorusE(Model):
    """
    Takuma Ebisu, Ryutaro Ichise: TorusE: Knowledge Graph Embedding on a Lie Group. AAAI 2018: 1819-1826.
    Takuma Ebisu, Ryutaro Ichise: Generalized Translation-Based Embedding of Knowledge Graph. IEEE Trans. Knowl. Data
        Eng. 32(5): 941-951 (2020).
    """
    def __init__(self, ent_total, rel_total, dim, variant="eL2"):
        """
            dim (int): Number of dimensions for embeddings
            variant can be either L1, L2 or eL2. Default: eL2 (best results are reported for L1 and eL2).
        """
        super(TorusE, self).__init__(ent_total, rel_total)
        self.dim = dim
        self.variant = variant

    def get_default_loss(self):
        # Eq. (3). The TKDE paper proposes a new loss function but the results in the experiments are quite similar.
        return 'margin'

    def get_score_sign(self):
        # It is a distance. They use norms even though our implementation does not have norms.
        return -1

    def initialize_model(self):
        # Like TransE. See Table 1. From the paper: "Regularization is not required, in contrast with the other
        #   embedding methods."
        self.create_embedding(self.dim, emb_type="entity", name="e")
        self.create_embedding(self.dim, emb_type="relation", name="r")

    def _calc(self, h, r, t):
        # From here: https://github.com/TakumaE/TorusE/blob/master/models.py
        # The extended version of the paper (TKDE) provides more details.
        # This is the mu torus operation.
        x = h + r
        # y is just t.
        y = t
        # Get the fractional part of x and y.
        x_frac, y_frac = x - torch.floor(x), y - torch.floor(y)

        # See Section 5.3 in TKDE.
        if self.variant == 'L1':
            d = torch.abs(x_frac - y_frac)
            scores = 2 * torch.sum(torch.minimum(d, 1 - d), dim=-1)
        elif self.variant == 'L2':
            d = torch.pow(x_frac - y_frac, 2)
            scores = 4 * torch.pow(torch.sum(torch.minimum(d, 1 - d), dim=-1), 2)
        elif self.variant == 'eL2':
            scores = torch.pow(torch.sum(2 - 2 * torch.cos(2 * math.pi * (x - y)), dim=-1), 2)/4

        return scores

    def return_score(self, is_predict=False):
        (head_emb, rel_emb, tail_emb) = self.current_batch

        h = head_emb["e"]
        t = tail_emb["e"]
        r = rel_emb["r"]

        return self._calc(h, r, t)
