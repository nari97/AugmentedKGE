import math
import torch
from Models.Model import Model


class RotateCT(Model):
    """
    Yao Dong, Lei Wang, Ji Xiang, Xiaobo Guo, Yuqiang Xie: RotateCT: Knowledge Graph Embedding by Rotation and
        Coordinate Transformation in Complex Space. COLING 2022: 4918-4932.
    """
    def __init__(self, ent_total, rel_total, dim, norm=1):
        """
            dim (int): Number of dimensions for embeddings
            norm (int): L1 or L2 norm. Default: 1. They follow RotatE, which has norm equal to 1.
        """
        super(RotateCT, self).__init__(ent_total, rel_total)
        self.dim = dim
        self.pnorm = norm

    def get_default_loss(self):
        # Eq. (4).
        return 'soft_margin'

    def get_score_sign(self):
        # It is a distance (norm).
        return -1

    def initialize_model(self):
        # From the paper: "Entity embeddings are uniformly initialized, and the phases of relation embeddings are
        #   uniformly initialized between −pi and pi."

        # From the paper: "In addition, we perform regularization on h and t to avoid overfitting."
        self.create_embedding(self.dim, emb_type="entity", name="e_real", reg=True)
        self.create_embedding(self.dim, emb_type="entity", name="e_img", reg=True)

        # Polar form is r*e^{i\theta}=r*(cos\theta + i*sin\theta). |r|=1 entails that we only use \theta.
        self.create_embedding(self.dim, emb_type="relation", name="r_phase",
                              init_method="uniform", init_params=[-math.pi, math.pi])

        # "Both the real and imaginary parts of each dimension in displacement b are initialized to zero."
        self.create_embedding(self.dim, emb_type="relation", name="b_real", init_method="uniform", init_params=[0, 0])
        self.create_embedding(self.dim, emb_type="relation", name="b_img", init_method="uniform", init_params=[0, 0])

    def _calc(self, h_real, h_img, r_phase, b_real, b_img, t_real, t_img):
        hc = torch.view_as_complex(torch.stack((h_real, h_img), dim=-1))
        tc = torch.view_as_complex(torch.stack((t_real, t_img), dim=-1))
        bc = torch.view_as_complex(torch.stack((b_real, b_img), dim=-1))
        rc = torch.polar(torch.ones_like(r_phase), r_phase)

        # Section 3.1.
        return torch.linalg.norm((hc - bc) * rc - (tc - bc), dim=-1, ord=self.pnorm)

    def return_score(self, is_predict=False):
        (head_emb, rel_emb, tail_emb) = self.current_batch

        h_real, h_img = head_emb["e_real"], head_emb["e_img"]
        t_real, t_img = tail_emb["e_real"], tail_emb["e_img"]
        r_phase, b_real, b_img = rel_emb["r_phase"], rel_emb["b_real"], rel_emb["b_img"]

        return self._calc(h_real, h_img, r_phase, b_real, b_img, t_real, t_img)
