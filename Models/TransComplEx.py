import math
import torch
from Models.Model import Model


class TransComplEx(Model):
    """
    Mojtaba Nayyeri, Chengjin Xu, Yadollah Yaghoobzadeh, Hamed Shariat Yazdi, Jens Lehmann: Toward Understanding The
        Effect Of Loss function On Then Performance Of Knowledge Graph Embedding. CoRR abs/1909.00519 (2019).
    """
    def __init__(self, ent_total, rel_total, dim, norm=1):
        """
            dim (int): Number of dimensions for embeddings
            norm (int): L1 or L2 norm. Default: 1.
        """
        super(TransComplEx, self).__init__(ent_total, rel_total)
        self.dim = dim
        self.pnorm = norm

    def get_default_loss(self):
        # Eq. (12).
        return 'soft_margin'

    def get_score_sign(self):
        # It is a distance (norm).
        return -1

    def initialize_model(self):
        # See Eq. (1).
        self.create_embedding(self.dim, emb_type="entity", name="e_real")
        self.create_embedding(self.dim, emb_type="entity", name="e_img")
        self.create_embedding(self.dim, emb_type="relation", name="r_real")
        self.create_embedding(self.dim, emb_type="relation", name="r_img")

    def _calc(self, h_real, h_img, r_real, r_img, t_real, t_img):
        hc = torch.view_as_complex(torch.stack((h_real, h_img), dim=-1))
        tc = torch.view_as_complex(torch.stack((t_real, t_img), dim=-1))
        rc = torch.view_as_complex(torch.stack((r_real, r_img), dim=-1))
        # Eq. (1).
        return torch.linalg.norm(hc + rc - tc.conj(), dim=-1, ord=self.pnorm)

    def return_score(self, is_predict=False):
        (head_emb, rel_emb, tail_emb) = self.current_batch

        h_real, h_img = head_emb["e_real"], head_emb["e_img"]
        t_real, t_img = tail_emb["e_real"], tail_emb["e_img"]
        r_real, r_img = rel_emb["r_real"], rel_emb["r_img"]

        return self._calc(h_real, h_img, r_real, r_img, t_real, t_img)
