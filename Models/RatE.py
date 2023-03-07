import math
import torch
from Models.Model import Model


class RatE(Model):
    """
    Hao Huang, Guodong Long, Tao Shen, Jing Jiang, Chengqi Zhang: RatE: Relation-Adaptive Translating Embedding for
        Knowledge Graph Completion. COLING 2020: 556-567.
    """
    def __init__(self, ent_total, rel_total, dim, norm=1):
        """
            dim (int): Number of dimensions for embeddings
            norm (int): L1 or L2 norm. Default: 1 (Eq. (4)).
        """
        super(RatE, self).__init__(ent_total, rel_total)
        self.dim = dim
        self.pnorm = norm

    def get_default_loss(self):
        # Eq. (7).
        return 'soft_margin'

    def get_score_sign(self):
        # It is a distance (norm).
        return -1

    def initialize_model(self):
        # Section 2.3. Same embeddings as RotatE plus weights.
        self.create_embedding(self.dim, emb_type="entity", name="e_real")
        self.create_embedding(self.dim, emb_type="entity", name="e_img")
        # |r|=1 entails that the absolute part of r is 1.
        self.create_embedding(self.dim, emb_type="relation", name="r_phase",
                              init_method="uniform", init_params=[0, 2 * math.pi])
        # From the paper: "...where the weights are specified for each relation..."
        # There are eights weights in total. Regularization: see Eq. (7). By default L1.
        for component in ['1', '2', '3', '4', '5', '6', '7', '8']:
            self.create_embedding(1, emb_type="relation", name="wr_"+component, reg=True)

    def _calc(self, h_real, h_img, r_phase, wr, t_real, t_img):
        # Get polar form using phase.
        r = torch.polar(torch.ones_like(r_phase), r_phase)
        r_real, r_img = r.real, r.imag
        # Get the eight components of wr.
        (wr_1, wr_2, wr_3, wr_4, wr_5, wr_6, wr_7, wr_8) = wr
        # Get the products.
        ac, ad = h_real * r_real, h_real * r_img
        bc, bd = h_img * r_real, h_img * r_img
        # Use the weights to compute the weighted product (Eq. (2)).
        mult_real = wr_1 * ac + wr_2 * ad + wr_3 * bc + wr_4 * bd
        mult_img = wr_5 * ac + wr_6 * ad + wr_7 * bc + wr_8 * bd
        # Get as complex numbers.
        mult = torch.view_as_complex(torch.stack((mult_real, mult_img), dim=-1))
        tc = torch.view_as_complex(torch.stack((t_real, t_img), dim=-1))
        # Eq. (4).
        return torch.linalg.norm(mult - tc, dim=-1, ord=self.pnorm)

    def return_score(self, is_predict=False):
        (head_emb, rel_emb, tail_emb) = self.current_batch

        h_real, h_img = head_emb["e_real"], head_emb["e_img"]
        t_real, t_img = tail_emb["e_real"], tail_emb["e_img"]
        r_phase = rel_emb["r_phase"]
        wr = (rel_emb["wr_1"], rel_emb["wr_2"], rel_emb["wr_3"], rel_emb["wr_4"],
              rel_emb["wr_5"], rel_emb["wr_6"], rel_emb["wr_7"], rel_emb["wr_8"])

        return self._calc(h_real, h_img, r_phase, wr, t_real, t_img)
