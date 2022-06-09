import numpy as np
import torch
from Models.Model import Model


class Analogy(Model):
    def __init__(self, ent_total, rel_total, dim):
        """
        Args:
            ent_total (int): Total number of entities
            rel_total (int): Total number of relations
            dim (int): Dimensions for embeddings
        """
        super(Analogy, self).__init__(ent_total, rel_total)
        self.dim = dim

    def get_default_loss(self):
        return 'soft'

    def initialize_model(self):
        self.create_embedding(self.dim, emb_type="entity", name="e")
        self.create_embedding(self.dim, emb_type="relation", name="r")

        # Not mentioned in the original paper.
        #self.register_scale_constraint(emb_type="entity", name="e")

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
        return torch.sum(torch.bmm(self.get_matrix(r), h.view(-1, self.dim, 1)).view(-1, self.dim) * t, dim=1)

    def return_score(self, is_predict=False):
        (head_emb, rel_emb, tail_emb) = self.current_batch

        h = head_emb["e"]
        t = tail_emb["e"]
        r = rel_emb["r"]

        return self._calc(h, r, t)
