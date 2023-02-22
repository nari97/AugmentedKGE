import torch
from Models.Model import Model


class MDE(Model):
    """
    Afshin Sadeghi, Damien Graux, Hamed Shariat Yazdi, Jens Lehmann: MDE: Multiple Distance Embeddings for Link
        Prediction in Knowledge Graphs. ECAI 2020: 1427-1434.
    """
    def __init__(self, ent_total, rel_total, dim, norm=2):
        """
            dim (int): Number of dimensions for embeddings
            norm (int): L1 or L2 norm. Default: 2. "We use the value 2 for p in p-norm..."
        """
        super(MDE, self).__init__(ent_total, rel_total)
        self.dim = dim
        self.pnorm = norm

    def get_default_loss(self):
        # Eq. (6).
        # The implementation of this loss function (LimitLoss) only includes beta_2 (referred to as alpha) since it
        #   follows the TransEdge paper. Therefore, beta_1=1 in our case.
        return 'limit'

    def initialize_model(self):
        # See above Eq. (5).
        for c in ['i', 'j', 'k', 'l']:
            self.create_embedding(self.dim, emb_type="entity", name="e_"+c)
            self.create_embedding(self.dim, emb_type="relation", name="r_"+c)
            self.create_embedding(1, emb_type="global", name="w_"+c)
        # The paper mentions psi is a hyperparameter. We include it as a parameter.
        self.create_embedding(1, emb_type="global", name="psi")

    def si(self, w, h, r, t):
        # Eq. (1).
        return w*torch.linalg.norm(h + r - t, dim=-1, ord=self.pnorm)

    def sj(self, w, h, r, t):
        # Eq. (2).
        return w*torch.linalg.norm(t + r - h, dim=-1, ord=self.pnorm)

    def sk(self, w, h, r, t):
        # Eq. (3).
        return w*torch.linalg.norm(h + t - r, dim=-1, ord=self.pnorm)

    def sl(self, w, h, r, t):
        # Eq. (4).
        return w*torch.linalg.norm(h - r * t, dim=-1, ord=self.pnorm)

    def _calc(self, w, h, r, t, psi):
        (wi, wj, wk, wl) = w
        (hi, hj, hk, hl) = h
        (ri, rj, rk, rl) = r
        (ti, tj, tk, tl) = t
        # Eq. (5).
        return -(self.si(wi, hi, ri, ti) + self.sj(wj, hj, rj, tj) + self.sk(wk, hk, rk, tk) + \
                    self.sl(wl, hl, rl, tl) - psi)

    def return_score(self, is_predict=False):
        (head_emb, rel_emb, tail_emb) = self.current_batch

        (hi, hj, hk, hl) = (head_emb["e_i"], head_emb["e_j"], head_emb["e_k"], head_emb["e_l"])
        (ti, tj, tk, tl) = (tail_emb["e_i"], tail_emb["e_j"], tail_emb["e_k"], tail_emb["e_l"])
        (ri, rj, rk, rl) = (rel_emb["r_i"], rel_emb["r_j"], rel_emb["r_k"], rel_emb["r_l"])
        (wi, wj, wk, wl) = (self.current_global_embeddings["w_i"], self.current_global_embeddings["w_j"],
                                self.current_global_embeddings["w_k"], self.current_global_embeddings["w_l"])
        psi = self.current_global_embeddings["psi"]

        return self._calc((wi, wj, wk, wl), (hi, hj, hk, hl), (ri, rj, rk, rl), (ti, tj, tk, tl), psi)
