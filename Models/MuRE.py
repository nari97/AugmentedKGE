import math
import torch
from Models.Model import Model

# Check this: https://github.com/ibalazevic/multirelational-poincare/blob/master/model.py
class MuRE(Model):

    def __init__(self, ent_total, rel_total, dim):
        super(MuRE, self).__init__(ent_total, rel_total)
        self.dim = dim

    def get_default_loss(self):
        return 'bce'

    def initialize_model(self):
        self.create_embedding(self.dim, emb_type="entity", name="e")
        self.create_embedding(1, emb_type="entity", name="bh")
        self.create_embedding(1, emb_type="entity", name="bt")
        self.create_embedding(self.dim, emb_type="relation", name="r")
        self.create_embedding(self.dim, emb_type="relation", name="R")

    def _calc(self, h, bh, r, R, t, bt):
        return -torch.linalg.norm(R*h - t+r, dim=-1, ord=2) + bh.flatten() + bt.flatten()

    def return_score(self, is_predict=False):
        (head_emb, rel_emb, tail_emb) = self.current_batch

        h = head_emb["e"]
        bh = head_emb["bh"]
        t = tail_emb["e"]
        bt = tail_emb["bt"]
        r, R = rel_emb["r"], rel_emb["R"]

        return self._calc(h, bh, r, R, t, bt)
