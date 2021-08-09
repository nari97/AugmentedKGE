import torch
from .Model import Model
from Utils.Embedding import Embedding
import torch.nn.functional as F

class RotatE(Model):

    def __init__(self, ent_total, rel_total, dims, norm = 2):
        super(RotatE, self).__init__(ent_total, rel_total)

        self.dims = dims
        self.norm = norm

        self.entities = Embedding(self.ent_tot, 2*self.dims)
        self.relations = Embedding(self.rel_tot, self.dims, init = 'uniform', init_params=[0, 2*self.pi_const.item()])

        #print (self.entities.emb)
        #print (self.relations.emb)

    def normalize(self):
        self.entities.normalize()
        self.relations.normalize()
        
    def multiply(self, a, b):
        #print (a.shape)
        #print (b.shape)
        x = a[:,:,0]
        y = a[:,:,1]
        u = b[:,:,0]
        v = b[:,:,1]

        a = x*u - y*v
        b = x*v + y*u
        c = torch.stack((a,b), dim = -1)
      
        return c

    def _calc(self, h, r, t):
        
        th = h.view(h.shape[0], self.dims, -1)
        tt = t.view(t.shape[0], self.dims, -1)      
      
        
        real = torch.cos(r)
        img = torch.sin(r)
        #print (real.shape, img.shape)
        tr = torch.stack((real, img), dim = -1)
        
        #print (th.shape, tr.shape)
        inner = self.multiply(th, tr) - tt
        inner = inner.reshape(inner.shape[0], -1)

        data = -torch.norm(inner, dim = -1, p = 2)

        return data

    def forward(self, data):

        batch_h = self.get_batch(data, "h")
        batch_r = self.get_batch(data, "r")
        batch_t = self.get_batch(data, "t")

        h = self.entities.get_embedding(batch_h)
        r = self.relations.get_embedding(batch_r)
        t = self.entities.get_embedding(batch_t)

        score = self._calc(h,r,t).flatten()

        return score

    def predict(self, data):
        score = -self.forward(data)

        return score



