import torch
from .Model import Model
from Utils.Embedding import Embedding
import torch.nn.functional as F
from Utils.utils import normalize

class RotatE(Model):

    def __init__(self, ent_total, rel_total, dims, norm = 2, inner_norm = False):
        super(RotatE, self).__init__(ent_total, rel_total)

        self.dims = dims

        self.dim_e = self.dims*2
        self.dim_r = self.dims 

        self.norm = norm
        self.inner_norm = inner_norm
        self.model_name = "rotate"
        self.pi_const = torch.Tensor([3.14159265358979323846])
        
        norm_params = {"p" : 2, "dim" : -1, "maxnorm" : 1}

        self.create_embedding(self.ent_tot, self.dim_e, emb_type = "entity", name = "e", normMethod = "norm", norm_params = norm_params, init = "uniform", init_params=[0, 1])
        
        self.create_embedding(self.rel_tot, self.dim_r, emb_type = "relation", name = "r", normMethod = "norm", norm_params= norm_params, init = "uniform", init_params=[0, 2*self.pi_const.item()])

        self.register_params()


    def normalize_inner(self, h, r, t):
        h = normalize(h, dim = -1, p = 2)
        r = normalize(r, dim = -1, p = 2)
        t = normalize(t, dim = -1, p = 2)

        return h, r, t
        
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

        tr = torch.stack((real, img), dim = -1)
        
        #print (th.shape, tr.shape)
        inner = self.multiply(th, tr) - tt
        inner = inner.reshape(inner.shape[0], -1)

        data = -torch.norm(inner, dim = -1, p = 2)

        return data

    def returnScore(self, head_emb, rel_emb, tail_emb):

        h = head_emb["e"]
        t = tail_emb["e"]
        r = rel_emb["r"]

        if self.inner_norm:
            h, r, t = self.normalize_inner(h, r, t)

        score = self._calc(h,r,t).flatten()

        return score




