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

        self.entities = Embedding(self.ent_tot, self.dim_e)
        self.relations = Embedding(self.rel_tot, self.dim_r)
        self.ent_transfer = Embedding(self.ent_tot, self.dim_e)
        self.rel_transfer = Embedding(self.rel_tot, self.dim_r)

    def normalize(self):
        self.entities.normalize(norm = 'clamp', maxnorm = 1)
        self.relations.normalize(norm = 'clamp', maxnorm = 1)

    def normalize_inner(self, h, r, t):
        h = clamp_norm(h, p = 2, dim = -1, maxnorm = 1)
        t = clamp_norm(t, p = 2, dim = -1, maxnorm = 1)
        r = clamp_norm(r, p = 2, dim = -1, maxnorm = 1)

        return h, t, r


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

    def forward(self, data):

        batch_h = self.get_batch(data, "h")
        batch_r = self.get_batch(data, "r")
        batch_t = self.get_batch(data, "t")

        h = self.entities.get_embedding(batch_h)
        r = self.relations.get_embedding(batch_r)
        t = self.entities.get_embedding(batch_t)

        if self.inner_nrom:
            h, r, t = self.normalize_inner(h, r, t)

        h_transfer = self.entities.get_embedding(batch_h)
        r_transfer = self.relations.get_embedding(batch_r)
        t_transfer = self.entities.get_embedding(batch_t)

        h = self._transfer(h, h_transfer, r_transfer)
        t = self._transfer(t, t_transfer, r_transfer)

        score = self._calc(h,r,t).flatten()

        return score

    def predict(self, data):
        score = -self.forward(data)
        return score



