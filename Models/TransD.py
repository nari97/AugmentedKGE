import torch

from Models.Model import Model
from Utils.Embedding import Embedding
from Utils.utils import clamp_norm, normalize
import torch.nn.functional as F

class TransD(Model):
    """
    TransD :cite:`ji2015knowledge` is a translation-based embedding approach that introduces the concept that entity and relation embeddings are no longer represented in the same space. Entity embeddings are represented in space :math:`\mathbb{R}^{k}` and relation embeddings are represented in space :math:`\mathbb{R}^{d}` where :math:`k \geq d`.TransD also introduces additional embeddings :math:`\mathbf{w_{h}}, \mathbf{w_{t}} \in \mathbb{R}^{k}` and :math:`\mathbf{w_{r} \in \mathbb{R}^{d}}`. I is the identity matrix.
    The scoring function for TransD is defined as
    
    :math:`f_{r}(h,t) = -||h_{\\bot} + \mathbf{r} - t_{\\bot}||`
    
    :math:`h_{\\bot} = (\mathbf{w_{r}}\mathbf{w_{h}^{T}} + I^{d \\times k})\,\mathbf{h}`
    
    :math:`t_{\\bot} = (\mathbf{w_{r}}\mathbf{w_{t}^{T}} + I^{d \\times k})\,\mathbf{t}`

    TransD imposes contraints like :math:`||\mathbf{h}||_{2} \leq 1, ||\mathbf{t}||_{2} \leq 1, ||\mathbf{r}||_{2} \leq 1, ||h_{\\bot}||_{2} \leq 1` and :math:`||t_{\\bot}||_{2} \leq 1`
    """

    def __init__(self, ent_total, rel_total, dim_e, dim_r, norm = 2, inner_norm = False):
        """
        Args:
            ent_total (int): Total number of entities
            rel_total (int): Total number of relations
            dim_e (int): Number of dimensions for entity embeddings
            dim_r (int): Number of dimensions for relation embeddings
            norm (int): L1 or L2 norm. Default: 2
        """
        super(TransD, self).__init__(ent_total, rel_total)

        self.dim_e = dim_e
        self.dim_r = dim_r
        self.norm = norm
        self.inner_norm = inner_norm
        self.model_name = "transd"
        norm_params = {"p" : 2, "dim" : -1, "maxnorm" : 1}

        self.entities = self.create_embedding(self.ent_tot, self.dim_e, emb_type = "entity", name = "e", normMethod = "clamp", norm_params = norm_params)
        
        self.relations = self.create_embedding(self.rel_tot, self.dim_r, emb_type = "relation", name = "r", normMethod = "clamp", norm_params= norm_params)
        
        self.ent_transfer = self.create_embedding(self.ent_tot, self.dim_e, emb_type = "entity", name = "e_t", normMethod = "clamp", norm_params = norm_params)
        
        self.rel_transfer = self.create_embedding(self.rel_tot, self.dim_r, emb_type = "relation", name = "r_t", normMethod = "none", norm_params= norm_params)
        self.register_params()
        
    def normalize_inner(self, h,r, t):
        h = clamp_norm(h, p = 2, dim = -1, maxnorm = 1)
        t = clamp_norm(t, p = 2, dim = -1, maxnorm = 1)
        r = clamp_norm(r, p = 2, dim = -1, maxnorm = 1)
    
        
        return h,r, t


    def _calc(self, h,r,t):
        score = h + r - t
        answer = -torch.pow(torch.norm(score, 2, -1),2)
        
        return answer

    def _transfer(self, e, e_transfer, r_transfer):
        
        wh = e_transfer
        wr = r_transfer

        m = wr*torch.sum(wh*e, dim=-1, keepdim=True)
        I = torch.matmul(e, torch.eye(e.shape[1], m.shape[1]))

        return clamp_norm(m+I, p=2, dim=-1, maxnorm = 1)

    def returnScore(self, head_emb, rel_emb, tail_emb):

        h = head_emb["e"]
        t = tail_emb["e"]
        r = rel_emb["r"]

        h_transfer = head_emb["e_t"]
        t_transfer = tail_emb["e_t"]
        r_transfer = rel_emb["r_t"]

        
        if self.inner_norm:
            h,r, t = self.normalize_inner(h,r, t)

        h = self._transfer(h, h_transfer, r_transfer)
        t = self._transfer(t, t_transfer, r_transfer)
        score = self._calc(h,r,t).flatten()

        return score




