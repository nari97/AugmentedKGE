import math
import torch
from Models.Model import Model


class RotatE(Model):
    """
    Zhiqing Sun, Zhi-Hong Deng, Jian-Yun Nie, Jian Tang: RotatE: Knowledge Graph Embedding by Relational Rotation in
        Complex Space. ICLR (Poster) 2019.
    """
    def __init__(self, ent_total, rel_total, dim, norm=1):
        """
            dim (int): Number of dimensions for embeddings
            norm (int): L1 or L2 norm. Default: 1. See footnote 2 in the paper.
        """
        super(RotatE, self).__init__(ent_total, rel_total)
        self.dim = dim
        self.pnorm = norm

    def get_default_loss(self):
        # Eq. (4).
        return 'soft_margin'

    def initialize_model(self):
        # See Eq. (1).
        self.create_embedding(self.dim, emb_type="entity", name="e_real")
        self.create_embedding(self.dim, emb_type="entity", name="e_img")
        # Polar form is r*e^{i\theta}=r*(cos\theta + i*sin\theta). |r|=1 entails that we only use \theta.
        # From the paper: "the phases of the relation embeddings are uniformly initialized between 0 and 2*pi."
        self.create_embedding(self.dim, emb_type="relation", name="r_phase",
                              init_method="uniform", init_params=[0, 2 * math.pi])

        # From the paper: "No regularization is used since we find that the fixed margin prevent our model from
        #   over-fitting."

    def _calc(self, h_real, h_img, r_phase, t_real, t_img):
        hc = torch.view_as_complex(torch.stack((h_real, h_img), dim=-1))
        tc = torch.view_as_complex(torch.stack((t_real, t_img), dim=-1))
        rc = torch.view_as_complex(torch.stack((torch.cos(r_phase), torch.sin(r_phase)), dim=-1))

        # Eq. (3).
        return -torch.linalg.norm(hc * rc - tc, dim=-1, ord=self.pnorm)

    def return_score(self, is_predict=False):
        (head_emb, rel_emb, tail_emb) = self.current_batch

        h_real, h_img = head_emb["e_real"], head_emb["e_img"]
        t_real, t_img = tail_emb["e_real"], tail_emb["e_img"]
        r_phase = rel_emb["r_phase"]

        return self._calc(h_real, h_img, r_phase, t_real, t_img)
