import math
import torch
from Models.Model import Model


class RotPro(Model):
    """
    Tengwei Song, Jie Luo, Lei Huang: Rot-Pro: Modeling Transitivity by Projection in Knowledge Graph Embedding.
        NeurIPS 2021: 24695-24706.
    """
    def __init__(self, ent_total, rel_total, dim, norm=1):
        """
            dim (int): Number of dimensions for embeddings
            norm (int): L1 or L2 norm. Default: 1 (like RotatE).
        """
        super(RotPro, self).__init__(ent_total, rel_total)
        self.dim = dim
        self.pnorm = norm

    def get_default_loss(self):
        # Eq. (9).
        return 'soft_margin'

    def get_score_sign(self):
        # It is a distance (norm).
        return -1

    def initialize_model(self):
        # Parameters like RotatE.
        # From the paper: "Both the real and imaginary parts of the entity embeddings are uniformly initialized..."
        self.create_embedding(self.dim, emb_type="entity", name="e_real")
        self.create_embedding(self.dim, emb_type="entity", name="e_img")
        self.create_embedding(self.dim, emb_type="relation", name="r_phase",
                              init_method="uniform", init_params=[-math.pi/2, math.pi/2],
                              # Make phases of relations between [-pi, pi]
                              # See: https://github.com/tewiSong/Rot-Pro/blob/main/codes/model.py#L326
                              # There is also a discussion about intialization in the paper.
                              # It seems the best results were achieved using -pi/2 and pi/2.
                              norm_method="rescaling", norm_params={"a": -math.pi/2, "b": math.pi/2})

        # These are two projections to be used in pr.
        # See: https://github.com/tewiSong/Rot-Pro/blob/main/codes/model.py#L63
        self.create_embedding(self.dim, emb_type="relation", name="proj_a",
                              init_method="uniform", init_params=[.5, .5],
                              # The paper proposes a penalty added to the loss function to scale proj_a and proj_b
                              #     between 0 and 1 (see Section 3.4 and Eq. (10)). We use rescaling.
                              norm_method="rescaling", norm_params={"a": 0, "b": 1})
        self.create_embedding(self.dim, emb_type="relation", name="proj_b",
                              init_method="uniform", init_params=[.5, .5],
                              norm_method="rescaling", norm_params={"a": 0, "b": 1})
        self.create_embedding(self.dim, emb_type="relation", name="proj_phase",
                              init_method="uniform", init_params=[0, 0])

    def _calc(self, h_real, h_img, r_phase, proj_a, proj_b, proj_phase, t_real, t_img):
        # Get these in polar form.
        r_phase, proj_phase = torch.polar(torch.ones_like(r_phase), r_phase), \
            torch.polar(torch.ones_like(proj_phase), proj_phase)

        def rotation(c, r):
            # Eq. (1).
            (re, im) = c
            return r.real * re + -r.imag * im, r.imag * re + r.real * im

        # Eq. (3).
        def pr(re, im):
            # See: https://github.com/tewiSong/Rot-Pro/blob/main/codes/model.py#L340
            return re * ma + im * mb, re * mb + im * md

        # Compute ma, mb, md. See: https://github.com/tewiSong/Rot-Pro/blob/main/codes/model.py#L337
        ma = torch.pow(proj_phase.real, 2) * proj_a + torch.pow(proj_phase.imag, 2) * proj_b
        mb = proj_phase.real * proj_phase.imag * (proj_b - proj_a)
        md = torch.pow(proj_phase.real, 2) * proj_b + torch.pow(proj_phase.imag, 2) * proj_a

        # Equation 8.
        hr = torch.view_as_complex(torch.stack(rotation(pr(h_real, h_img), r_phase), dim=-1))
        tt = torch.view_as_complex(torch.stack(pr(t_real, t_img), dim=-1))
        return torch.linalg.norm(hr - tt, dim=-1, ord=self.pnorm)

    def return_score(self, is_predict=False):
        (head_emb, rel_emb, tail_emb) = self.current_batch

        h_real, h_img = head_emb["e_real"], head_emb["e_img"]
        t_real, t_img = tail_emb["e_real"], tail_emb["e_img"]
        r_phase, proj_a, proj_b, proj_phase = rel_emb["r_phase"], \
            rel_emb["proj_a"], rel_emb["proj_b"], rel_emb["proj_phase"]

        return self._calc(h_real, h_img, r_phase, proj_a, proj_b, proj_phase, t_real, t_img)
