import torch
from .Model import Model
from Utils.Embedding import Embedding
import torch.nn.functional as F

class TransD(Model):

    def __init__(self, ent_total, rel_total, dim_e, dim_r, norm = 2):
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

        self.entities = Embedding(self.ent_tot, self.dim_e)
        self.relations = Embedding(self.rel_tot, self.dim_r)
        self.ent_transfer = Embedding(self.ent_tot, self.dim_e)
        self.rel_transfer = Embedding(self.rel_tot, self.dim_r)

    def normalize(self):
        self.entities.normalize()
        self.relations.normalize()

    def _calc(self, h,r,t):
        score = h + r - t
        answer = -torch.pow(torch.norm(score, 2, -1),2)
        
        return answer

    def _transfer(self, e, e_transfer, r_transfer):
        
        wh = e_transfer
        wr = r_transfer

        m = wr*torch.sum(wh*e, dim=-1, keepdim=True)
        I = torch.matmul(e, torch.eye(e.shape[1], m.shape[1]))

        return F.normalize(m+I, p=2, dim=-1)

    def forward(self, data):

        batch_h = self.get_batch(data, "h")
        batch_r = self.get_batch(data, "r")
        batch_t = self.get_batch(data, "t")

        h = self.entities.get_embedding(batch_h)
        r = self.relations.get_embedding(batch_r)
        t = self.entities.get_embedding(batch_t)

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



