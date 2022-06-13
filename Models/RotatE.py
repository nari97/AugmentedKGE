import math
import torch
from .Model import Model


class RotatE(Model):

    def __init__(self, ent_total, rel_total, dim, norm=1):
        super(RotatE, self).__init__(ent_total, rel_total)
        self.dim = dim
        self.pnorm = norm

    def get_default_loss(self):
        return 'soft_margin'

    def initialize_model(self):
        self.create_embedding(self.dim, emb_type="entity", name="e_real")
        self.create_embedding(self.dim, emb_type="entity", name="e_img")
        # |r|=1 entails that the absolute part of r is 1.
        self.create_embedding(self.dim, emb_type="relation", name="r_phase",
                              init="uniform", init_params=[0, 2 * math.pi])

    def _calc(self, h_real, h_img, r_phase, t_real, t_img):
        hc = torch.view_as_complex(torch.stack((h_real, h_img), dim=-1))
        tc = torch.view_as_complex(torch.stack((t_real, t_img), dim=-1))
        rc = torch.view_as_complex(torch.stack((torch.cos(r_phase), torch.sin(r_phase)), dim=-1))
        return -torch.linalg.norm(hc * rc - tc, dim=-1, ord=self.pnorm)

    def return_score(self, is_predict=False):
        (head_emb, rel_emb, tail_emb) = self.current_batch

        h_real, h_img = head_emb["e_real"], head_emb["e_img"]
        t_real, t_img = tail_emb["e_real"], tail_emb["e_img"]
        r_phase = rel_emb["r_phase"]

        return self._calc(h_real, h_img, r_phase, t_real, t_img)
