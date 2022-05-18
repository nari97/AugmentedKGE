import torch
from Models.Model import Model


class TransE(Model):
    """
    TransE :cite:`bordes2013translating` is a representative relational distance model. All the embeddings, entities and relations are both represented in the same space :math:`\mathbb{R}^{d}` where d is the dimension of the embedding. Given a triple (head, relation, tail), transE imposes the constraint that :math:`h+r \\approx t`.
    The scoring function for TransE is defined as 

    :math:`f_{r}(h,t) = -||\mathbf{h}+\mathbf{r}-\mathbf{t}||_{1/2}`

    TransE enforces additional constraints :math:`||\mathbf{h}||_{2} = 1` and :math:`||\mathbf{t}||_{2} = 1`.
    """
    def __init__(self, ent_total, rel_total, dim, norm=2):
        """
        Args:
            ent_total (int): Total number of entities
            rel_total (int): Total number of relations
            dims (int): Number of dimensions for embeddings
            norm (int): L1 or L2 norm. Default: 2
        """
        super(TransE, self).__init__(ent_total, rel_total)
        self.dim = dim
        self.pnorm = norm

    def initialize_model(self):
        self.create_embedding(self.dim, emb_type="entity", name="e", norm_method="norm")
        self.create_embedding(self.dim, emb_type="relation", name="r")

    def _calc(self, h, r, t):
        return -torch.linalg.norm(h + r - t, dim=-1, ord=self.pnorm)

    def return_score(self, head_emb, rel_emb, tail_emb, is_predict=False):
        h = head_emb["e"]
        t = tail_emb["e"]
        r = rel_emb["r"]

        return self._calc(h, r, t)
