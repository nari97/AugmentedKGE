import torch
from .Model import Model


class HolE(Model):
    """
    Maximilian Nickel, Lorenzo Rosasco, Tomaso A. Poggio: Holographic Embeddings of Knowledge Graphs. AAAI 2016:
        1955-1961.
    """
    def __init__(self, ent_total, rel_total, dim):
        """
            dim (int): Number of dimensions for embeddings
        """
        super(HolE, self).__init__(ent_total, rel_total)
        self.dim = dim

    def get_default_loss(self):
        # In Experiments: "...we reimplemented all models and used the identical... ranking loss of Eq. (3)."
        return 'margin'

    def initialize_model(self):
        # Eq. (12). In Eq. (3) of the loss function no regularization is used.
        self.create_embedding(self.dim, emb_type="entity", name="e")
        self.create_embedding(self.dim, emb_type="relation", name="r")
        
    def _calc(self, h, r, t):
        # Last term in Eq. (12) says e_t(r * e_h), where a*b=F-1(F(a) x F(b)), where x is the Hadamard product (see the
        #   end of that section).
        fh = torch.fft.rfft(h, dim=-1)
        fr = torch.fft.rfft(r, dim=-1)
        return torch.sum(t * torch.fft.irfft(fr * fh, dim=-1, n=self.dim), dim=-1)

    def return_score(self, is_predict=False):
        (head_emb, rel_emb, tail_emb) = self.current_batch

        h = head_emb["e"]
        t = tail_emb["e"]
        r = rel_emb["r"]

        return self._calc(h, r, t)



