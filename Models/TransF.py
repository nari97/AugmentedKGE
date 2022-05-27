import torch
from Models.Model import Model


class TransF(Model):
    # The authors propose their modification over several models; we selected TransE.
    def __init__(self, ent_total, rel_total, dim):
        super(TransF, self).__init__(ent_total, rel_total)
        self.dim = dim

    def get_default_loss(self):
        return 'margin'

    def initialize_model(self):
        self.create_embedding(self.dim, emb_type="entity", name="e", norm_method="norm")
        self.create_embedding(self.dim, emb_type="relation", name="r")

    def _calc(self, h, r, t):
        return torch.sum((h+r)*t + h*(t-r), -1)

    def return_score(self, is_predict=False):
        (head_emb, rel_emb, tail_emb) = self.current_batch

        h = head_emb["e"]
        t = tail_emb["e"]
        r = rel_emb["r"]

        return self._calc(h, r, t)
