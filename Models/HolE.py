import torch
from .Model import Model


class HolE(Model):

    def __init__(self, ent_total, rel_total, dims, use_gpu):
        super(HolE, self).__init__(ent_total, rel_total, dims, "hole", use_gpu)

        self.create_embedding(self.dims, emb_type="entity", name="e")
        self.create_embedding(self.dims, emb_type="relation", name="r")
    
        self.register_scale_constraint(emb_type="entity", name="e", p=2)
        self.register_scale_constraint(emb_type="relation", name="r", p=2)
        
    def _calc(self, h, r, t):
        # Last term in Eq. 13 says e_t(r * e_h), where a*b=F-1(F(a) x F(b)), where x is the Hadamard product.
        fh = torch.fft.rfft(h, dim=-1)
        fr = torch.fft.rfft(r, dim=-1)
        return torch.sum(t * torch.fft.irfft(fr * fh, dim=-1, n=r.shape[-1]), dim=-1, keepdim=False)

    def return_score(self, head_emb, rel_emb, tail_emb, is_predict=False):
        h = head_emb["e"]
        t = tail_emb["e"]
        r = rel_emb["r"]

        return self._calc(h, r, t)



