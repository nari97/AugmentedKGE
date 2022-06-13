import math
import torch
from .Model import Model


class RotPro(Model):

    def __init__(self, ent_total, rel_total, dim, norm=1):
        super(RotPro, self).__init__(ent_total, rel_total)
        self.dim = dim
        self.pnorm = norm

    def get_default_loss(self):
        return 'soft_margin'

    def initialize_model(self):
        self.create_embedding(self.dim, emb_type="entity", name="e_real")
        self.create_embedding(self.dim, emb_type="entity", name="e_img")

        self.create_embedding(self.dim, emb_type="relation", name="r_phase",
                              init="uniform", init_params=[-math.pi, math.pi],
                              # Make phases of relations between [-pi, pi]
                              norm_method="rescaling", norm_params={"a": -math.pi, "b": math.pi})
        self.create_embedding(self.dim, emb_type="relation", name="a",
                              # https://github.com/tewiSong/Rot-Pro/blob/main/codes/model.py#L63
                              init="uniform", init_params=[.5, .5])
        self.create_embedding(self.dim, emb_type="relation", name="b",
                              init="uniform", init_params=[.5, .5])
        self.create_embedding(self.dim, emb_type="relation", name="p")

        self.register_custom_constraint(self.penalty_constraint)

    # Equation 10.
    def penalty_constraint(self, head_emb, rel_emb, tail_emb, alpha=.25):
        return (self.apply_penalty(rel_emb["a"]) + self.apply_penalty(rel_emb["b"])) * alpha

    # Check: https://github.com/tewiSong/Rot-Pro/blob/main/codes/model.py#L420
    def apply_penalty(self, x, gamma=.00005, beta=1.5):
        x1 = x - 1.0
        x0 = x - 0.0
        penalty = torch.ones_like(x)
        penalty[torch.abs(x1 * x0) > gamma] = beta
        return torch.linalg.norm(x1 * x * penalty, dim=-1, ord=2)

    # Equation 1.
    def rotation(self, c, phase):
        (re, im) = c
        return torch.cos(phase) * re + -torch.sin(phase) * im, \
                torch.sin(phase) * re + torch.cos(phase) * im

    # Equation 3.
    def pr(self, re, im, a, b, phase):
        cos, sin = torch.cos(phase), torch.sin(phase)

        # Equation 2.
        mr_a = a * cos * cos + b * sin * sin
        mr_b = cos * sin * (-a + b)
        mr_d = a * sin * sin + b * cos * cos

        return re * mr_a + im * mr_b, re * mr_b + im * mr_d

    def _calc(self, h_real, h_img, r_phase, a, b, p, t_real, t_img):
        # Equation 8.
        hr = torch.view_as_complex(torch.stack(self.rotation(self.pr(h_real, h_img, a, b, p), r_phase), dim=-1))
        tt = torch.view_as_complex(torch.stack(self.pr(t_real, t_img, a, b, p), dim=-1))

        return -torch.linalg.norm(hr - tt, dim=-1, ord=self.pnorm)

    def return_score(self, is_predict=False):
        (head_emb, rel_emb, tail_emb) = self.current_batch

        h_real, h_img = head_emb["e_real"], head_emb["e_img"]
        t_real, t_img = tail_emb["e_real"], tail_emb["e_img"]
        r_phase, a, b, p = rel_emb["r_phase"], rel_emb["a"], rel_emb["b"], rel_emb["p"]

        return self._calc(h_real, h_img, r_phase, a, b, p, t_real, t_img)
