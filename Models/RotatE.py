import math
import torch
from .Model import Model

class RotatE(Model):

    def __init__(self, ent_total, rel_total, dims, norm=1):
        super(RotatE, self).__init__(ent_total, rel_total, dims, "rotate")

        self.pnorm = norm
        self.create_embedding(self.dims, emb_type="entity", name="e_real")
        self.create_embedding(self.dims, emb_type="entity", name="e_img")
        self.create_embedding(self.dims, emb_type="relation", name="r_abs")
        self.create_embedding(self.dims, emb_type="relation", name="r_phase",
                              init="uniform", init_params=[0, 2 * math.pi])

    def _calc(self, h_real, h_img, t_real, t_img, r_abs, r_phase):
        hc = torch.view_as_complex(torch.stack((h_real, h_img), dim=-1))
        tc = torch.view_as_complex(torch.stack((t_real, t_img), dim=-1))
        rc = torch.view_as_complex(torch.stack((r_abs*torch.cos(r_phase), r_abs*torch.sin(r_phase)), dim=-1))
        return -torch.linalg.norm(hc * rc - tc, dim=-1, ord=self.pnorm)

    def return_score(self, head_emb, rel_emb, tail_emb, is_predict=False):
        h_real = head_emb["e_real"]
        h_img = head_emb["e_img"]
        t_real = tail_emb["e_real"]
        t_img = tail_emb["e_img"]

        r_abs = rel_emb["r_abs"]
        r_phase = rel_emb["r_phase"]

        return self._calc(h_real, h_img, t_real, t_img, r_abs, r_phase)
