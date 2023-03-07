import torch
from Models.Model import Model


class TransMS(Model):
    """
    Shihui Yang, Jidong Tian, Honglun Zhang, Junchi Yan, Hao He, Yaohui Jin: TransMS: Knowledge Graph Embedding for
        Complex Relations by Multidirectional Semantics. IJCAI 2019: 1935-1942.
    """
    def __init__(self, ent_total, rel_total, dim, norm=2):
        """
            dim (int): Number of dimensions for embeddings. Even though the paper talks about two dimensions, in the
                        end, they are both the same. From the paper: "...ke = kr... in our model."
            norm (int): L1 or L2 norm. Default: 2
        """
        super(TransMS, self).__init__(ent_total, rel_total)
        self.dim = dim
        self.pnorm = norm

    def get_default_loss(self):
        # Eq. (5).
        return 'margin'

    def get_score_sign(self):
        # It is a distance (norm).
        return -1

    def initialize_model(self):
        # See Section 3.2.
        self.create_embedding(self.dim, emb_type="entity", name="e")
        self.create_embedding(self.dim, emb_type="relation", name="r")
        self.create_embedding(1, emb_type="relation", name="alpha")
        # See Section 3.3.
        self.register_scale_constraint(emb_type="entity", name="e")
        self.register_scale_constraint(emb_type="relation", name="r")

    def _calc(self, h, r, alpha, t):
        # Eq. (4).
        return torch.linalg.norm(-torch.tanh(t*r)*h + r + alpha*(h*t) - torch.tanh(h*r)*t, dim=-1, ord=self.pnorm)

    def return_score(self, is_predict=False):
        (head_emb, rel_emb, tail_emb) = self.current_batch

        h = head_emb["e"]
        t = tail_emb["e"]
        r, alpha = rel_emb["r"], rel_emb["alpha"]

        return self._calc(h, r, alpha, t)
