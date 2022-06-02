import math
import torch
from .Model import Model


class pRotatE(Model):

    def __init__(self, ent_total, rel_total, dim, norm=1, c=.5):
        super(pRotatE, self).__init__(ent_total, rel_total)
        self.dim = dim
        self.pnorm = norm
        self.c = c

    def get_default_loss(self):
        return 'soft_margin'

    def initialize_model(self):
        self.create_embedding(self.dim, emb_type="entity", name="e_phase",
                              init="uniform", init_params=[0, 2 * math.pi])
        self.create_embedding(self.dim, emb_type="relation", name="r_phase",
                              init="uniform", init_params=[0, 2 * math.pi])

    def _calc(self, h_phase, r_phase, t_phase):
        return -2*self.c*torch.linalg.norm(torch.sin((h_phase+r_phase-t_phase)/2), dim=-1, ord=self.pnorm)

    def return_score(self, is_predict=False):
        (head_emb, rel_emb, tail_emb) = self.current_batch

        h_phase = head_emb["e_phase"]
        t_phase = tail_emb["e_phase"]
        r_phase = rel_emb["r_phase"]

        return self._calc(h_phase, r_phase, t_phase)
