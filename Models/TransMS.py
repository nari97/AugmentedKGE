import torch
from Models.Model import Model


class TransMS(Model):
    def __init__(self, ent_total, rel_total, dim, norm=2):
        super(TransMS, self).__init__(ent_total, rel_total)
        self.dim = dim
        self.pnorm = norm

    def get_default_loss(self):
        return 'margin'

    def initialize_model(self):
        self.create_embedding(self.dim, emb_type="entity", name="e")
        self.create_embedding(self.dim, emb_type="relation", name="r")
        self.create_embedding(1, emb_type="relation", name="alpha")

        self.register_scale_constraint(emb_type="entity", name="e")
        self.register_scale_constraint(emb_type="relation", name="r")

    def _calc(self, h, r, alpha, t):
        return -torch.linalg.norm(-torch.tanh(t*r)*h + r + alpha*(h*t) - torch.tanh(h*r)*t, dim=-1, ord=self.pnorm)

    def return_score(self, is_predict=False):
        (head_emb, rel_emb, tail_emb) = self.current_batch

        h = head_emb["e"]
        t = tail_emb["e"]
        r = rel_emb["r"]
        alpha = rel_emb["alpha"]

        return self._calc(h, r, alpha, t)
