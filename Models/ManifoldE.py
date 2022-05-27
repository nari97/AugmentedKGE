import torch
from Models.Model import Model


class ManifoldE(Model):

    def __init__(self, ent_total, rel_total, dim):
        super(ManifoldE, self).__init__(ent_total, rel_total)
        self.dim = dim

    def get_default_loss(self):
        return 'margin'

    def initialize_model(self):
        self.create_embedding(self.dim, emb_type="entity", name="e")
        self.create_embedding(self.dim, emb_type="relation", name="rh")
        self.create_embedding(self.dim, emb_type="relation", name="rt")

    def _calc(self, h, rh, t, rt):
        # Assuming linear kernel:
        return -torch.sum((h+rh) * (t+rt), -1)

    def return_score(self, is_predict=False):
        (head_emb, rel_emb, tail_emb) = self.current_batch

        h = head_emb["e"]
        t = tail_emb["e"]
        rh, rt = rel_emb["rh"], rel_emb["rt"]

        return self._calc(h, rh, t, rt)
