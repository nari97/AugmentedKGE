import math
import torch
from .Model import Model


class HARotatE(Model):

    def __init__(self, ent_total, rel_total, dim, norm=1, m=2):
        super(HARotatE, self).__init__(ent_total, rel_total)
        self.dim = dim
        self.pnorm = norm
        self.m = m

    def get_default_loss(self):
        return 'soft_margin'

    def initialize_model(self):
        self.create_embedding(self.dim, emb_type="entity", name="e_real")
        self.create_embedding(self.dim, emb_type="entity", name="e_img")
        # |r|=1 entails that the absolute part of r is 1.
        self.create_embedding(self.dim, emb_type="relation", name="r_phase",
                              init="uniform", init_params=[0, 2 * math.pi])
        self.create_embedding(self.m, emb_type="relation", name="w", init="uniform", init_params=[-2, 2])

    def _calc(self, h_real, h_img, r_phase, w, t_real, t_img):
        w_expand = torch.repeat_interleave(w, self.m, dim=1)

        if w_expand.shape[1] != self.dim:
            w_expand = torch.cat((w_expand, w_expand[:, w_expand.shape[1]-1:]), dim=1)

        hc = torch.view_as_complex(torch.stack((w_expand * h_real, w_expand * h_img), dim=-1))
        tc = torch.view_as_complex(torch.stack((t_real, t_img), dim=-1))
        rc = torch.view_as_complex(torch.stack((torch.cos(r_phase), torch.sin(r_phase)), dim=-1))

        return -torch.linalg.norm(hc * rc - tc, dim=-1, ord=self.pnorm)

    def return_score(self, is_predict=False):
        (head_emb, rel_emb, tail_emb) = self.current_batch

        h_real, h_img = head_emb["e_real"], head_emb["e_img"]
        t_real, t_img = tail_emb["e_real"], tail_emb["e_img"]
        r_phase, w = rel_emb["r_phase"], rel_emb["w"]

        return self._calc(h_real, h_img, r_phase, w, t_real, t_img)
