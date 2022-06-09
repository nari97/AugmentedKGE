import torch
from Models.Model import Model


class TransEDT(Model):
    def __init__(self, ent_total, rel_total, dim, norm=2):
        super(TransEDT, self).__init__(ent_total, rel_total)
        self.dim = dim
        self.pnorm = norm

    def get_default_loss(self):
        return 'margin'

    def initialize_model(self):
        self.create_embedding(self.dim, emb_type="entity", name="e", norm_method="norm")
        self.create_embedding(self.dim, emb_type="relation", name="r")
        self.create_embedding(self.dim, emb_type="entity", name="ehalpha")
        self.create_embedding(self.dim, emb_type="entity", name="etalpha")
        self.create_embedding(self.dim, emb_type="relation", name="ralpha")

        self.register_scale_constraint(emb_type="entity", name="ehalpha", p=2, z=.3)
        self.register_scale_constraint(emb_type="entity", name="etalpha", p=2, z=.3)
        self.register_scale_constraint(emb_type="relation", name="ralpha", p=2, z=.3)

    def _calc(self, h, halpha, r, ralpha, t, talpha):
        return -torch.linalg.norm((h + halpha) + (r + ralpha) - (t + talpha), dim=-1, ord=self.pnorm)

    def return_score(self, is_predict=False):
        (head_emb, rel_emb, tail_emb) = self.current_batch

        h, halpha = head_emb["e"], head_emb["ehalpha"]
        t, talpha = tail_emb["e"], tail_emb["etalpha"]
        r, ralpha = rel_emb["r"], rel_emb["ralpha"]

        return self._calc(h, halpha, r, ralpha, t, talpha)
