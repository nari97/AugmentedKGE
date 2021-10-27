import torch
from Models.Model import Model
from Utils.Embedding import Embedding
import torch.nn.functional as F
from Utils.utils import normalize

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

        self.ent_tot = ent_total
        self.rel_tot = rel_total
        self.dims = dims
        self.norm = norm
        self.inner_norm = inner_norm

        #Create normalisation parameters
        norm_params = {"p" : 2, "dim" : -1, "maxnorm" : 1}

        self.create_embedding(self.ent_tot, self.dims, emb_type = "entity", name = "e", normMethod = "norm", norm_params = norm_params)
        
        self.create_embedding(self.rel_tot, self.dims, emb_type = "relation", name = "r", normMethod = "none", norm_params= norm_params)
        

    def normalize_inner(self, h, t):
        
        h = normalize(h, p = 2, dim = -1)
        t = normalize(t, p = 2, dim = -1)
        
        return h,t

    def _calc(self, h,r,t):
        return -torch.norm(h+r-t, dim = -1, p = self.norm)

    def returnScore(self, head_emb, rel_emb, tail_emb):
        h = head_emb["e"]
        t = tail_emb["e"]
        r = rel_emb["r"]
        
        if self.inner_norm:
            h,t = self.normalize_inner(h,t)

        return self._calc(h, r, t)


