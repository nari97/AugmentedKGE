import torch
from .Model import Model
from Utils.Embedding import Embedding
from Utils.utils import clamp_norm, normalize
import torch.nn.functional as F

class HolE(Model):

    def __init__(self, ent_total, rel_total, dims, norm = 2, inner_norm = False):
        super(HolE, self).__init__(ent_total, rel_total)

        self.dims = dims
        self.norm = norm
        self.inner_norm = inner_norm
        self.model_name = "hole"
        norm_params = {"p" : 2, "dim" : -1, "maxnorm" : 1}

        self.create_embedding(self.ent_tot, self.dims, emb_type = "entity", name = "e", normMethod = "clamp", norm_params = norm_params)
        
        self.create_embedding(self.rel_tot, self.dims, emb_type = "relation", name = "r", normMethod = "clamp", norm_params= norm_params)
    
        self.register_params()
        
    def normalize_inner(self, h, r, t):
        h = clamp_norm(h, dim = -1, p = 2, maxnorm = 1)
        r = clamp_norm(r, dim = -1, p = 2, maxnorm = 1)
        t = clamp_norm(t, dim = -1, p = 2, maxnorm = 1)
        return h,r,t


    def _calc(self, h,r,t):
        fourierH = torch.fft.rfft(h, dim = -1)
        fourierT = torch.fft.rfft(t, dim = -1)
      
        conjH = torch.conj(fourierH)
      
        inv = torch.fft.irfft(conjH*fourierT, dim = -1)
      
        if r.shape[1]>inv.shape[1]:
            r = r[:, :inv.shape[1]]
        elif inv.shape[1]>r.shape[1]:
            inv = inv[:, :r.shape[1]]
       
        answer = torch.sum(r*inv, dim = -1)
      
        return answer

    def returnScore(self, head_emb, rel_emb, tail_emb):

        h = head_emb["e"]
        t = tail_emb["e"]
        r = rel_emb["r"]

        if self.inner_norm:
            h,r,t = self.normalize_inner(h,r,t)

        score = self._calc(h,r,t).flatten()

        return score



