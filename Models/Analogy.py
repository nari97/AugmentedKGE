import numpy as np
import torch
from Models.Model import Model


class Analogy(Model):
    """
    Hanxiao Liu, Yuexin Wu, Yiming Yang: Analogical Inference for Multi-relational Embeddings. ICML 2017: 2168-2178.
    """
    def __init__(self, ent_total, rel_total, dim):
        """
            dim (int): Dimensions for embeddings
        """
        super(Analogy, self).__init__(ent_total, rel_total)
        self.dim = dim

    def get_default_loss(self):
        # Section 6.4.1.
        # -log sig(y \phi) = -log 1 + log(1+exp(-y \phi) = log(1+exp(-y \phi), which is SoftMarginLoss
        return 'soft'

    def initialize_model(self):
        # Section 2.1.
        self.create_embedding(self.dim, emb_type="entity", name="e")
        # Section 2.1 mentions that relation embeddings are matrices. However, Corollary 4.2.1 reformulates the problem
        #   using almost diagonal matrices. So this embedding is also a vector.
        # From the paper: "the alternative optimization problem can be handled by simply binding together the
        #   coefficients within each of those 2x2 blocks in Br. Note that each Br consists of only m free parameters",
        #   where m is the embedding dimension.
        self.create_embedding(self.dim, emb_type="relation", name="r")

    def get_matrix(self, r):
        # r = [a, b, c, d, e]
        # r as a block-diagonal matrix:
        #   a, b, 0, 0, 0
        #  -b, a, 0, 0, 0
        #   0, 0, c, d, 0
        #   0, 0,-d, c, 0
        #   0, 0, 0, 0, e
        even_indexes = torch.LongTensor(np.arange(0, self.dim, 2))
        odd_indexes = torch.LongTensor(np.arange(1, self.dim, 2))

        r_diag = r[:, even_indexes].repeat_interleave(2, dim=1)

        if r_diag.shape[1] != self.dim:
            # Get rid of the last position.
            r_diag = r_diag[:,:self.dim]

        r_u = r[:, odd_indexes].repeat_interleave(2, dim=1)
        # Get rid of the last position.
        r_u = r_u[:,:self.dim - 1]

        r_l = r_u*-1

        return torch.diag_embed(r_diag) + torch.diag_embed(r_u, offset=1) + torch.diag_embed(r_l, offset=-1)

    def _calc(self, h, r, t):
        # h*r-t (Eq. (1))
        return torch.sum(torch.bmm(self.get_matrix(r), h.view(-1, self.dim, 1)).view(-1, self.dim) - t, dim=1)

    def return_score(self, is_predict=False):
        (head_emb, rel_emb, tail_emb) = self.current_batch

        h = head_emb["e"]
        t = tail_emb["e"]
        r = rel_emb["r"]

        return self._calc(h, r, t)
