import math
import torch
from Models.Model import Model


class FiveStarE(Model):
    """
    Mojtaba Nayyeri, Sahar Vahdati, Can Aykul, Jens Lehmann: 5* Knowledge Graph Embeddings with Projective
        Transformations. AAAI 2021: 9064-9072.
    """
    def __init__(self, ent_total, rel_total, dim):
        """
            dim (int): Number of dimensions for embeddings
        """
        super(FiveStarE, self).__init__(ent_total, rel_total)
        self.dim = dim

    def get_default_loss(self):
        # "...applied 1-N scoring loss..."
        # In their code, they use CrossEntropyLoss: https://github.com/mojtabanayyeri/5-StartE/blob/5-StarE/kbc/optimizers.py#L30
        return 'logsoftmax'

    def get_score_sign(self):
        # It is a similarity.
        return 1

    def initialize_model(self):
        # From the paper: "...applied 1-N scoring loss with N3 regularization..."

        self.create_embedding(self.dim, emb_type="entity", name="e_real", reg=True)
        self.create_embedding(self.dim, emb_type="entity", name="e_img", reg=True)

        # Model Formulation section.
        for component in ['a', 'b', 'c', 'd']:
            self.create_embedding(self.dim, emb_type="relation", name="r_real_"+component, reg=True)
            self.create_embedding(self.dim, emb_type="relation", name="r_img_" + component, reg=True)

    def _calc(self, h_real, h_img, r_real_a, r_img_a, r_real_b, r_img_b,
                          r_real_c, r_img_c, r_real_d, r_img_d, t_real, t_img):
        hc = torch.view_as_complex(torch.stack((h_real, h_img), dim=-1))
        tc = torch.view_as_complex(torch.stack((t_real, t_img), dim=-1))
        rca = torch.view_as_complex(torch.stack((r_real_a, r_img_a), dim=-1))
        rcb = torch.view_as_complex(torch.stack((r_real_b, r_img_b), dim=-1))
        rcc = torch.view_as_complex(torch.stack((r_real_c, r_img_c), dim=-1))
        rcd = torch.view_as_complex(torch.stack((r_real_d, r_img_d), dim=-1))

        # Eq. (4).
        hr = (rca * hc + rcb) / (rcc * hc + rcd)

        # Eq. (6).
        return torch.sum(torch.mul(hr, torch.conj(tc)).real, dim=-1)

    def return_score(self, is_predict=False):
        (head_emb, rel_emb, tail_emb) = self.current_batch

        h_real, h_img = head_emb["e_real"], head_emb["e_img"]
        t_real, t_img = tail_emb["e_real"], tail_emb["e_img"]
        r_real_a, r_img_a, r_real_b, r_img_b, r_real_c, r_img_c, r_real_d, r_img_d = \
            rel_emb["r_real_a"], rel_emb["r_img_a"], rel_emb["r_real_b"], rel_emb["r_img_b"], \
            rel_emb["r_real_c"], rel_emb["r_img_c"], rel_emb["r_real_d"], rel_emb["r_img_d"]

        return self._calc(h_real, h_img, r_real_a, r_img_a, r_real_b, r_img_b,
                          r_real_c, r_img_c, r_real_d, r_img_d, t_real, t_img)
