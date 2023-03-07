import math
import torch
from Models.Model import Model


class HARotatE(Model):
    """
    Shensi Wang, Kun Fu, Xian Sun, Zequn Zhang, Shuchao Li, Li Jin: Hierarchical-aware relation rotational knowledge
        graph embedding for link prediction. Neurocomputing 458: 259-270 (2021).
    """
    def __init__(self, ent_total, rel_total, dim, norm=1, mp=None):
        """
            dim (int): Number of dimensions for embeddings
            norm (int): L1 or L2 norm. Default: 1 as in RotatE but it is a hyperparameter in the paper.
            mp (float): .5 <= m < 0. Number of divisions of embeddings. It is not specified as a percentage; however,
                            it indicates the number of "divisions" of dim.
        """
        super(HARotatE, self).__init__(ent_total, rel_total)
        self.dim = dim
        self.pnorm = norm
        if mp is None or mp > .5 or mp <= 0:
            # This is not stated in the paper; however, Table 3 seems to be dependent on the number of relations.
            mp = .5
        self.m = int(math.floor(self.dim * mp))

    def get_default_loss(self):
        # Eq. (6).
        return 'soft_margin'

    def get_score_sign(self):
        # It is a distance (norm).
        return -1

    def initialize_model(self):
        self.create_embedding(self.dim, emb_type="entity", name="e_real")
        self.create_embedding(self.dim, emb_type="entity", name="e_img")
        # Polar form is r*e^{i\theta}=r*(cos\theta + i*sin\theta). |r|=1 entails that we only use \theta.
        # From the paper: "[we] constrain the phases of the relation embeddings between 0 and 2*pi."
        self.create_embedding(self.dim, emb_type="relation", name="r_phase",
                              init_method="uniform", init_params=[0, 2 * math.pi],
                              norm_method="rescaling", norm_params={"a": 0, "b": 2 * math.pi})
        # Section 5.3.
        self.create_embedding(self.m, emb_type="relation", name="w", init_method="uniform", init_params=[-2, 2])

    def _calc(self, h_real, h_img, r_phase, w, t_real, t_img):
        # we expand w so the multiplication works. See Eq. (3). We repeat ceil(|h| * m).
        w_expand = torch.repeat_interleave(w, int(math.ceil(self.dim / self.m)), dim=1)
        # If the shape is still not correct, cut until last dimension fits.
        if w_expand.shape[1] != self.dim:
            # w_expand[:, :self.dim] keeps only the initial dimensions until dim.
            w_expand = w_expand[:, :self.dim]

        hc = torch.view_as_complex(torch.stack((w_expand * h_real, w_expand * h_img), dim=-1))
        tc = torch.view_as_complex(torch.stack((t_real, t_img), dim=-1))
        rc = torch.polar(torch.ones_like(r_phase), r_phase)

        # Eq. (4).
        return torch.linalg.norm(hc * rc - tc, dim=-1, ord=self.pnorm)

    def return_score(self, is_predict=False):
        (head_emb, rel_emb, tail_emb) = self.current_batch

        h_real, h_img = head_emb["e_real"], head_emb["e_img"]
        t_real, t_img = tail_emb["e_real"], tail_emb["e_img"]
        r_phase, w = rel_emb["r_phase"], rel_emb["w"]

        return self._calc(h_real, h_img, r_phase, w, t_real, t_img)
