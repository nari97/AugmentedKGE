import math
import torch
from Models.Model import Model


class TimE(Model):
    """
    Qianjin Zhang, Ronggui Wang, Juan Yang, Lixia Xue: Knowledge graph embedding by translating in time domain space for
        link prediction. Knowl. Based Syst. 212: 106564 (2021)
    """
    def __init__(self, ent_total, rel_total, dim, norm=2):
        """
            dim (int): Number of dimensions for embeddings
            norm (int): L1 or L2 norm. Default: 2.
        """
        super(TimE, self).__init__(ent_total, rel_total)
        self.dim = dim
        self.pnorm = norm

    def get_default_loss(self):
        # Eq. (9).
        return 'soft_margin'

    def get_score_sign(self):
        # It is a distance (norm).
        return -1

    def initialize_model(self):
        self.create_embedding(self.dim, emb_type="entity", name="e")
        self.create_embedding(self.dim, emb_type="relation", name="r")
        self.create_embedding(self.dim, emb_type="relation", name="theta_h",
                              init_method="uniform", init_params=[0, 2 * math.pi])
        self.create_embedding(self.dim, emb_type="relation", name="theta_t",
                              init_method="uniform", init_params=[0, 2 * math.pi])
        # See Training protocol in Section 4.2.
        self.create_embedding(1, emb_type="global", name="w11")
        self.create_embedding(1, emb_type="global", name="w12")
        self.create_embedding(1, emb_type="global", name="w21")
        self.create_embedding(1, emb_type="global", name="w22")
        self.create_embedding(1, emb_type="global", name="lambda1")
        self.create_embedding(1, emb_type="global", name="lambda2")

    def _calc(self, h, theta_h, r, t, theta_t, w11, w12, w21, w22, lambda1, lambda2):
        def transfer(e, theta, w1, w2):
            # Eqs. (7) and (8).
            return e * (torch.cos(w1 * theta) + w2 * theta)

        # Eqs. (2), (5) and (6).
        return lambda1 * torch.linalg.norm(h + r - transfer(t, theta_t, w11, w12), dim=-1, ord=self.pnorm) + \
            lambda2 * torch.linalg.norm(t - r - transfer(h, theta_h, w21, w22), dim=-1, ord=self.pnorm)

    def return_score(self, is_predict=False):
        (head_emb, rel_emb, tail_emb) = self.current_batch

        h = head_emb["e"]
        t = tail_emb["e"]
        r, theta_h, theta_t = rel_emb["r"], rel_emb["theta_h"], rel_emb["theta_t"]

        w11, w12, w21, w22 = self.current_global_embeddings["w11"], self.current_global_embeddings["w12"], \
            self.current_global_embeddings["w21"], self.current_global_embeddings["w22"]
        lambda1, lambda2 = self.current_global_embeddings["lambda1"], self.current_global_embeddings["lambda2"]

        return self._calc(h, theta_h, r, t, theta_t, w11, w12, w21, w22, lambda1, lambda2)
