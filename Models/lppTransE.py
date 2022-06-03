import torch
from Models.Model import Model


class lppTransE(Model):
    def __init__(self, ent_total, rel_total, dim, norm=2):
        super(lppTransE, self).__init__(ent_total, rel_total)
        self.dim = dim
        self.pnorm = norm

    def get_default_loss(self):
        return 'margin'

    def initialize_model(self):
        self.create_embedding(self.dim, emb_type="entity", name="e", norm_method="norm")
        self.create_embedding(self.dim, emb_type="relation", name="r")
        self.create_embedding((self.dim, self.dim), emb_type="relation", name="mh")
        self.create_embedding((self.dim, self.dim), emb_type="relation", name="mt")

    def _calc(self, h, mh, r, t, mt):
        batch_size = h.shape[0]

        mmh = torch.matmul(mh, h.view(batch_size, -1, 1)).view(batch_size, self.dim)
        mmt = torch.matmul(mt, t.view(batch_size, -1, 1)).view(batch_size, self.dim)

        return -torch.linalg.norm(mmh + r - mmt, dim=-1, ord=self.pnorm)

    def return_score(self, is_predict=False):
        (head_emb, rel_emb, tail_emb) = self.current_batch

        h = head_emb["e"]
        t = tail_emb["e"]
        r, mh, mt = rel_emb["r"], rel_emb["mh"], rel_emb["mt"]

        return self._calc(h, mh, r, t, mt)
