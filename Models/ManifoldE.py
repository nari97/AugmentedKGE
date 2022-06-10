import torch
from Models.Model import Model


class ManifoldE(Model):

    def __init__(self, ent_total, rel_total, dim, norm=1):
        super(ManifoldE, self).__init__(ent_total, rel_total)
        self.dim = dim
        self.pnorm = norm

    def get_default_loss(self):
        return 'margin'

    def initialize_model(self):
        self.create_embedding(self.dim, emb_type="entity", name="e")
        self.create_embedding(self.dim, emb_type="relation", name="rh")
        self.create_embedding(self.dim, emb_type="relation", name="rt")
        self.create_embedding(1, emb_type="relation", name="dr")

    def _calc(self, h, rh, t, rt, dr):
        # Assuming linear kernel:
        return -torch.pow(torch.linalg.norm((h + rh) * (t + rt) - torch.pow(dr, 2), dim=-1, ord=self.pnorm), 2)

    def return_score(self, is_predict=False):
        (head_emb, rel_emb, tail_emb) = self.current_batch

        h = head_emb["e"]
        t = tail_emb["e"]
        rh, rt, dr = rel_emb["rh"], rel_emb["rt"], rel_emb["dr"]

        return self._calc(h, rh, t, rt, dr)
