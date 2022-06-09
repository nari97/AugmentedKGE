import torch
from Models.Model import Model


class MAKR(Model):

    def __init__(self, ent_total, rel_total, dim, norm=2):
        super(MAKR, self).__init__(ent_total, rel_total)
        self.dim = dim
        self.pnorm = norm

    def get_default_loss(self):
        return 'margin'

    def initialize_model(self):
        self.create_embedding(self.dim, emb_type="entity", name="e")
        self.create_embedding(self.dim, emb_type="relation", name="rh")
        self.create_embedding(self.dim, emb_type="relation", name="rt")
        self.create_embedding(1, emb_type="relation", name="dr")

        self.register_scale_constraint(emb_type="entity", name="e")
        self.register_scale_constraint(emb_type="relation", name="rh")
        self.register_scale_constraint(emb_type="relation", name="rt")

    def _calc(self, h, rh, t, rt, dr):
        return -torch.linalg.norm(rh * h - rt * t - torch.pow(dr, 2), dim=-1, ord=self.pnorm)

    def return_score(self, is_predict=False):
        (head_emb, rel_emb, tail_emb) = self.current_batch

        h = head_emb["e"]
        t = tail_emb["e"]
        rh, rt, dr = rel_emb["rh"], rel_emb["rt"], rel_emb["dr"]

        return self._calc(h, rh, t, rt, dr)
