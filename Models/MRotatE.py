import math
import torch
from .Model import Model


class MRotatE(Model):
    """
    Xuqian Huang, Jiuyang Tang, Zhen Tan, Weixin Zeng, Ji Wang, Xiang Zhao: Knowledge graph embedding by relational and
        entity rotation. Knowl. Based Syst. 229: 107310 (2021).
    """
    def __init__(self, ent_total, rel_total, dim, norm=1):
        """
            dim (int): Number of dimensions for embeddings
            norm (int): L1 or L2 norm. Default: 1, even though ||.|| is never defined in the paper.
        """
        super(MRotatE, self).__init__(ent_total, rel_total)
        self.dim = dim
        self.pnorm = norm

    def get_default_loss(self):
        # Eq. (7).
        return 'soft_margin'

    def initialize_model(self):
        # See Eq. (1). Similar to RotatE.
        self.create_embedding(self.dim, emb_type="entity", name="e_real")
        self.create_embedding(self.dim, emb_type="entity", name="e_img")
        # As in RotatE.
        self.create_embedding(self.dim, emb_type="relation", name="r_phase",
                              init_method="uniform", init_params=[0, 2 * math.pi])

        # See Eq. (3).
        self.create_embedding(self.dim, emb_type="entity", name="e")
        self.create_embedding(self.dim, emb_type="relation", name="r")

        # See Eq. (6). Omega is hyperparameter ("under the best weights obtained by the validation set training"). We
        #   consider it as a parameter. Omega is always between zero and one.
        self.create_embedding(1, emb_type="global", name="omega", init_method="uniform", init_params=[0, 1],
                              norm_method="rescaling", norm_params={"a": 0, "b": 1})

    def _calc(self, h_real, h_img, h, r_phase, r, t_real, t_img, t, omega):
        # This is RotatE (Eq. (2)).
        hc = torch.view_as_complex(torch.stack((h_real, h_img), dim=-1))
        tc = torch.view_as_complex(torch.stack((t_real, t_img), dim=-1))
        rc = torch.view_as_complex(torch.stack((torch.cos(r_phase), torch.sin(r_phase)), dim=-1))
        relation_rotation_scores = torch.linalg.norm(hc * rc - tc, dim=-1, ord=self.pnorm)

        # Eq. (4).
        entity_rotation_scores = torch.linalg.norm(h - t, dim=-1, ord=self.pnorm) - \
                                 torch.linalg.norm(r, dim=-1, ord=self.pnorm)

        # Eq. (6). Note that omega + phi = 1, so phi = 1 - omega.
        return -omega * relation_rotation_scores - (1 - omega) * entity_rotation_scores

    def return_score(self, is_predict=False):
        (head_emb, rel_emb, tail_emb) = self.current_batch

        h, h_real, h_img = head_emb["e"], head_emb["e_real"], head_emb["e_img"]
        t, t_real, t_img = tail_emb["e"], tail_emb["e_real"], tail_emb["e_img"]
        r, r_phase = rel_emb["r"], rel_emb["r_phase"]

        omega = self.current_global_embeddings["omega"].item()

        return self._calc(h_real, h_img, h, r_phase, r, t_real, t_img, t, omega)
