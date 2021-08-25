import torch
from Models.Model import Model
from Utils.Embedding import Embedding
import torch.nn.functional as F

class TransE(Model):
    """
    TransE :cite:`bordes2013translating` is a representative relational distance model. All the embeddings, entities and relations are both represented in the same space :math:`\mathbb{R}^{d}` where d is the dimension of the embedding. Given a triple (head, relation, tail), transE imposes the constraint that :math:`h+r \\approx t`.
    The scoring function for TransE is defined as 
    
    :math:`f_{r}(h,t) = -||\mathbf{h}+\mathbf{r}-\mathbf{t}||_{1/2}`
    
    TransE enforces additional constraints :math:`||\mathbf{h}||_{2} = 1` and :math:`||\mathbf{t}||_{2} = 1`.

    """

    #TODO : Update documentation for inner

    def __init__(self, ent_total, rel_total, dims, norm = 2, inner_norm = False):
        """
        Args:
            ent_total (int): Total number of entities
            rel_total (int): Total number of relations
            dims (int): Number of dimensions for embeddings
            norm (int): L1 or L2 norm. Default: 2
        """
        super(TransE, self).__init__(ent_total, rel_total)

        self.dims = dims
        self.norm = norm
        self.inner_norm = inner_norm

        self.entities = Embedding(self.ent_tot, self.dims)
        self.relations = Embedding(self.rel_tot, self.dims)

    def normalize(self):
        self.entities.normalize()
        
    def normalize_inner(self, h, t):
        h = F.normalize(h, p = 2, dim = -1)
        t = F.normalize(t, p = 2, dim = -1)

        return h,t

    def _calc(self, h,r,t):
        return -torch.norm(h+r-t, dim = -1, p = self.norm)


    def forward(self, data):

        batch_h = self.get_batch(data, "h")
        batch_r = self.get_batch(data, "r")
        batch_t = self.get_batch(data, "t")

        h = self.entities.get_embedding(batch_h)
        r = self.relations.get_embedding(batch_r)
        t = self.entities.get_embedding(batch_t)

        if self.inner_norm:
            h,t = self.normalize_inner(h,t)

        score = self._calc(h,r,t).flatten()

        return score

    def predict(self, data):
        score = -self.forward(data)

        return score



