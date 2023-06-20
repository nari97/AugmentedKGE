import torch
from Models.Model import Model


class SEEK(Model):
    """
    Wentao Xu, Shun Zheng, Liang He, Bin Shao, Jian Yin, Tie-Yan Liu: SEEK: Segmented Embedding of Knowledge Graphs.
        ACL 2020: 3888-3897.
    """
    def __init__(self, ent_total, rel_total, dim, k=4):
        """
            dim (int): Number of dimensions for embeddings
            k (int): Number of segments to use. Default: 4 (see Figure 3; best choice between MRR and running time).
        """
        super(SEEK, self).__init__(ent_total, rel_total)
        self.dim = dim
        self.k = k
        self.n = self.dim//self.k

    def get_default_loss(self):
        # Eq. (5).
        return 'soft'

    def get_score_sign(self):
        # It is a similarity.
        return 1

    def initialize_model(self):
        # See Eq. (5) for regularization.
        self.create_embedding(self.dim, emb_type="entity", name="e", reg=True)
        self.create_embedding(self.dim, emb_type="relation", name="r", reg=True)
        
    def _calc(self, h, r, t):
        batch_size = h.shape[0]
        scores = torch.zeros(batch_size, dtype=h.dtype, layout=h.layout, device=h.device)

        # Eq. (4).
        for x in range(0, self.k):
            for y in range(0, self.k):
                # See below Eq. (3). If x is odd and x+y>=k, then sxy=-1; sxy=1 otherwise.
                sxy = -1 if x % 2 == 1 and x + y >= self.k else 1
                # See below Eq. (4). wxy=y if x is even; wxy=(x+y)%k otherwise.
                wxy = y if x % 2 == 0 else (x+y) % self.k
                # See Eq. (4): rx, hy, twxy.
                scores += sxy*torch.sum(r[:, x*self.n:(x+1)*self.n] * h[:, y*self.n:(y+1)*self.n] *
                                        t[:, wxy*self.n:(wxy+1)*self.n], -1)

        return scores

    def return_score(self, is_predict=False):
        (head_emb, rel_emb, tail_emb) = self.current_batch

        h = head_emb["e"]
        t = tail_emb["e"]
        r = rel_emb["r"]

        return self._calc(h, r, t)
