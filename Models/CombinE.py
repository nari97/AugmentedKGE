import math
import torch
from Models.Model import Model


class CombinE(Model):
    def __init__(self, ent_total, rel_total, dim, norm=2):
        super(CombinE, self).__init__(ent_total, rel_total)
        self.dim = dim
        self.pnorm = norm

    def get_default_loss(self):
        return 'margin'

    def initialize_model(self):
        self.create_embedding(self.dim, emb_type="entity", name="ep")
        self.create_embedding(self.dim, emb_type="entity", name="em")
        self.create_embedding(self.dim, emb_type="relation", name="rp")
        self.create_embedding(self.dim, emb_type="relation", name="rm")

        self.register_scale_constraint(emb_type="entity", name="ep")
        self.register_scale_constraint(emb_type="entity", name="em")
        self.register_scale_constraint(emb_type="relation", name="rp", z=math.sqrt(2))
        self.register_scale_constraint(emb_type="relation", name="rm")

    def _calc(self, hp, hm, rp, rm, tp, tm):
        return -torch.pow(torch.linalg.norm(hp + tp - rp, dim=-1, ord=self.pnorm), 2) + \
                    -torch.pow(torch.linalg.norm(hm - tm - rm, dim=-1, ord=self.pnorm), 2)

    def return_score(self, is_predict=False):
        (head_emb, rel_emb, tail_emb) = self.current_batch

        hp, hm = head_emb["ep"], head_emb["em"]
        tp, tm = tail_emb["ep"], tail_emb["em"]
        rp, rm = rel_emb["rp"], rel_emb["rm"]

        return self._calc(hp, hm, rp, rm, tp, tm)
