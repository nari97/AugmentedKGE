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
        #self.register_scale_constraint(emb_type="entity", name="e", p=2)

    def _calc(self, h, r, t):
        # We wish to multiply h times the block-diagonal matrix as follows:
        # h = [h1, h2, h3, h4, h5]
        # r = [a, b, c, d, e]
        # r as a block-diagonal matrix:
        #   a, b, 0, 0, 0
        #  -b, a, 0, 0, 0
        #   0, 0, c, d, 0
        #   0, 0,-d, c, 0
        #   0, 0, 0, 0, e
        # We split h and r in blocks of size 2 and compute the matrix multiplication in chunks.

        h_split = torch.split(h, 2, dim=1)
        r_split = torch.split(r, 2, dim=1)

        # This is to get -1 1 for each triple.
        mask = torch.tensor([-1, 1], device=h.device).repeat(h.size()[0], 1)

        hr_mul = []
        for i in range(0, len(h_split)):
            hs, rs = h_split[i], r_split[i]

            # If rs has two components, e.g., a b, we need to multiply a b and -b a, in that order.
            hr_mul.append(torch.sum(hs * rs, dim=1))
            if rs.size()[1] == 2:
                hr_mul.append(torch.sum(hs * mask * torch.fliplr(rs), dim=1))

        return torch.sum(torch.stack(hr_mul,dim=1) * t, dim=1)

    def return_score(self, is_predict=False):
        (head_emb, rel_emb, tail_emb) = self.current_batch

        h = head_emb["e"]
        t = tail_emb["e"]
        r = rel_emb["r"]

        return self._calc(h, r, t)
