import torch
from Models.Model import Model


class SE(Model):
    def __init__(self, ent_total, rel_total, dim, norm=1):
        super(SE, self).__init__(ent_total, rel_total)
        self.dim = dim
        self.pnorm = norm

    def get_default_loss(self):
        return 'margin'

    def initialize_model(self):
        self.create_embedding(self.dim, emb_type="entity", name="e", norm_method="norm")
        self.create_embedding((self.dim, self.dim), emb_type="relation", name="rh")
        self.create_embedding((self.dim, self.dim), emb_type="relation", name="rt")

    def _calc(self, h, rh, rt, t):
        hrh = torch.bmm(rh, h.view(-1, self.dim, 1)).view(-1, self.dim)
        trt = torch.bmm(rt, t.view(-1, self.dim, 1)).view(-1, self.dim)
        return -torch.linalg.norm(hrh - trt, dim=-1, ord=self.pnorm)

    def return_score(self, is_predict=False):
        (head_emb, rel_emb, tail_emb) = self.current_batch

        h = head_emb["e"]
        t = tail_emb["e"]
        rh, rt = rel_emb["rh"], rel_emb["rt"]

        return self._calc(h, rh, rt, t)
