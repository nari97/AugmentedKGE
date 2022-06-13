import math
import torch
from .Model import Model


class RatE(Model):

    def __init__(self, ent_total, rel_total, dim, norm=1):
        super(RatE, self).__init__(ent_total, rel_total)
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
        self.create_embedding(8, emb_type="relation", name="wr", reg=True)

    def _calc(self, h_real, h_img, r_phase, wr, t_real, t_img):
        r_real, r_img = torch.cos(r_phase), torch.sin(r_phase)
        ac, ad = h_real * r_real, h_real * r_img
        bc, bd = h_img * r_real, h_img * r_img

        # Use the weights to compute the weighted product.
        mult_real = wr[:, 0].view(-1, 1) * ac + wr[:, 1].view(-1, 1) * ad + wr[:, 2].view(-1, 1) * bc + \
                    wr[:, 3].view(-1, 1) * bd
        mult_img = wr[:, 4].view(-1, 1) * ac + wr[:, 5].view(-1, 1) * ad + wr[:, 6].view(-1, 1) * bc + \
                   wr[:, 7].view(-1, 1) * bd

        mult = torch.view_as_complex(torch.stack((mult_real, mult_img), dim=-1))
        tc = torch.view_as_complex(torch.stack((t_real, t_img), dim=-1))

        return -torch.linalg.norm(mult - tc, dim=-1, ord=self.pnorm)

    def return_score(self, is_predict=False):
        (head_emb, rel_emb, tail_emb) = self.current_batch

        h_real, h_img = head_emb["e_real"], head_emb["e_img"]
        t_real, t_img = tail_emb["e_real"], tail_emb["e_img"]
        r_phase, wr = rel_emb["r_phase"], rel_emb["wr"]

        return self._calc(h_real, h_img, r_phase, wr, t_real, t_img)
