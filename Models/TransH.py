import torch
from Models.Model import Model
from Utils.Embedding import Embedding
import torch.nn.functional as F

class TransH(Model):
    
    """
    TransH :cite:`wang2014knowledge` allows entities to have different representations for different relations by creating an additional embedding :math:`\mathbf{w_{r}} \in \mathbb{R}^{d}`.
    This is done by projecting entities onto a hyperplane specific to the relation r and with normal vector :math:`\mathbf{w_{r}}`.
   
    :math:`f_r(h,t) = -||h_{\\bot} + \\mathbf{r} -t_{\\bot}||_{2}^{2}` 
    
    :math:`h_{\\bot} = \\mathbf{h} - \\mathbf{w_{r}^{T}}\\mathbf{h} \\mathbf{w_{r}}`
    
    :math:`t_{\\bot} = \\mathbf{t} - \\mathbf{w_{r}^{T}} \\mathbf{t} \\mathbf{w_{r}}`
    
    
    TransH imposes additional constraints :math:`||\mathbf{h}||_{2} \leq 1`, :math:`||\mathbf{t}||_{2} \leq 1` and :math:`||\mathbf{w_{r}}|| = 1`.
    
    """

    def __init__(self, ent_total, rel_total, dims, norm = 2, inner_norm = False):
        """
        Args:
            ent_total (int): Total number of entities
            rel_total (int): Total number of relations
            dims (int): Number of dimensions for embeddings
            norm (int): L1 or L2 norm. Default: 2
        """

        super(TransH, self).__init__(ent_total, rel_total)

        self.dims = dims
        self.norm = norm
        self.inner_norm = inner_norm

        self.entities = Embedding(self.ent_tot, self.dims)
        self.relations = Embedding(self.rel_tot, self.dims)
        self.norm_vector = Embedding(self.rel_tot, self.dims)

    def normalize(self):
        self.entities.normalize()
        self.norm_vector.normalize()

    def normalize_inner(self, h, t, w_r):
        h = F.normalize(h, p = 2, dim = -1)
        t = F.normalize(t, p = 2, dim = -1)
        w_r = F.normalize(w_r, p = 2, dim = -1)

    def _calc(self, h, r, t, w_r):
        ht = h - torch.sum(h*w_r, dim = -1, keepdim = True).repeat(1, self.dims)*w_r
        tt = t - torch.sum(t*w_r, dim = -1, keepdim = True).repeat(1, self.dims)*w_r
        answer = -torch.pow(torch.norm(ht + r - tt, dim = -1, p = 2),2)

        return answer

    def forward(self, data):

        batch_h = self.get_batch(data, "h")
        batch_r = self.get_batch(data, "r")
        batch_t = self.get_batch(data, "t")

        

        h = self.entities.get_embedding(batch_h)
        r = self.relations.get_embedding(batch_r)
        t = self.entities.get_embedding(batch_t)
        w_r = self.norm_vector.get_embedding(batch_r)
        
        if self.inner_norm:
            h,t,w_r = self.normalize_inner(h, t, w_r)
            
        score = self._calc(h, r, t, w_r).flatten()

        return score

    def predict(self, data):
        score = -self.forward(data)

        return score



