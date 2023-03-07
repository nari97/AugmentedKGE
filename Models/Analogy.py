import numpy as np
import math
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

    def get_score_sign(self):
        # "Our goal is to learn ... such that ... gives high scores to valid triples, and low scores to invalid ones."
        return 1

    def initialize_model(self):
        # Section 2.1.
        self.create_embedding(self.dim, emb_type="entity", name="e")
        # Section 2.1 mentions that relation embeddings are matrices. However, Corollary 4.2.1 reformulates the problem
        #   using almost diagonal matrices. So this embedding is also a vector.
        # From the paper: "the alternative optimization problem can be handled by simply binding together the
        #   coefficients within each of those 2x2 blocks in Br. Note that each Br consists of only m free parameters",
        #   where m is the embedding dimension.
        self.create_embedding(self.dim, emb_type="relation", name="r")

    def _calc(self, h, r, t):
        # r = [r0, r1, r2, r3, r4]
        # r as a block-diagonal matrix:
        #   r0, r1,  0,  0, 0
        #  -r1, r0,  0,  0, 0
        #    0,  0, r2, r3, 0
        #    0,  0,-r3, r2, 0
        #    0,  0,  0,  0, r4
        # h = [h0, h1, h2, h3, h4]
        # h^T*r = [h0r0-h1r1, h1r0+h0r1, h2r2-h3r3, h3r2+h2r3, h4r4]
        # r_diag = [r0, r0, r2, r2, r4] (of the block-diagonal matrix)
        # h*r_diag gives all the first elements in h^T*r: [h0r0, h1r0, h2r2, h3r2, h4r4] => This is h_times_r
        # (Discarding the last position when dim is odd. We use even and odd indexes; note that we start in 0 (even).)
        # h_times_r[even] -= h[odd]*r[odd] (the even positions of the result)
        # h_times_r[odd] += h[even]*r[odd] (the odd positions of the result)

        batch_size = h.shape[0]
        # Even and odd indexes.
        even_indexes = torch.LongTensor(np.arange(0, self.dim, 2))
        odd_indexes = torch.LongTensor(np.arange(1, self.dim, 2))

        r_diag = r[:, even_indexes].repeat_interleave(2, dim=1)
        if r_diag.shape[1] != self.dim:
            # Get rid of the last position.
            r_diag = r_diag[:,:self.dim]
        h_times_r = h*r_diag

        if even_indexes.shape[0] % 2 != 0:
            # Get rid of the last position.
            even_indexes = even_indexes[:math.floor(self.dim/2)]

        h_times_r[:, even_indexes] -= h[:, odd_indexes] * r[:, odd_indexes]
        h_times_r[:, odd_indexes] += h[:, even_indexes] * r[:, odd_indexes]

        # h^T*B_r*t (Eq. (13))
        return torch.sum(h_times_r * t, dim=1)

    def return_score(self, is_predict=False):
        (head_emb, rel_emb, tail_emb) = self.current_batch

        h = head_emb["e"]
        t = tail_emb["e"]
        r = rel_emb["r"]

        return self._calc(h, r, t)
