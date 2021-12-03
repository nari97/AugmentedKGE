import torch
from Models.Model import Model
from Utils.Embedding import Embedding
import torch.nn.functional as F
from Utils.NormUtils import normalize, clamp_norm

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

        super(TransH, self).__init__(ent_total, rel_total, dims, "transh", inner_norm)

        self.entities = self.create_embedding(self.dims, emb_type = "entity", name = "e", normMethod = "clamp", norm_params = self.norm_params)
        self.relations = self.create_embedding(self.dims, emb_type = "relation", name = "r", normMethod = "none", norm_params= self.norm_params)
        self.w_relations = self.create_embedding(self.dims, emb_type = "relation", name = "w_r", normMethod = "norm", norm_params= self.norm_params)
        
        self.register_params()
        

    def _calc(self, h, r, t, w_r):
        ht = h - torch.sum(h*w_r, dim = -1, keepdim = True).repeat(1, self.dims)*w_r
        tt = t - torch.sum(t*w_r, dim = -1, keepdim = True).repeat(1, self.dims)*w_r
        answer = -torch.pow(torch.norm(ht + r - tt, dim = -1, p = 2),2)

        return answer

    def returnScore(self, head_emb, rel_emb, tail_emb):
        h = head_emb["e"]
        t = tail_emb["e"]
        r = rel_emb["r"]
        w_r = rel_emb["w_r"]
        

        
        return self._calc(h, r, t, w_r)


