import torch
from Models.Model import Model


class CyclE(Model):
    """
    Han Yang, Leilei Zhang, Bingning Wang, Ting Yao, Junfei Liu: Cycle or Minkowski: Which is More Appropriate for
        Knowledge Graph Embedding? CIKM 2021: 2301-2310.
    """
    def __init__(self, ent_total, rel_total, dim, norm=2):
        """
            dim (int): Number of dimensions for embeddings
            norm (int): L1 or L2 norm. Default: 2
        """
        super(CyclE, self).__init__(ent_total, rel_total)
        self.dim = dim
        self.pnorm = norm

    def get_default_loss(self):
        # After Eq. (8).
        return 'margin'

    def initialize_model(self):
        # From the paper: "CyclE replace[s] Minkowski in [T]ransE with Cycle metric, which continues the simplicity of
        #   [T]ransE without adding additional parameter[s]." So we just mimic TransE parameters and normalizations.
        self.create_embedding(self.dim, emb_type="entity", name="e", norm_method="norm")
        self.create_embedding(self.dim, emb_type="relation", name="r")
        # See Eq. (8). a, b, g and w correspond to A, b, \gamma and \omega in the paper. It is unclear the dimension of
        #   these; however, Section 7.1 reports best values for these parameters, and they are all single real numbers.
        #   Also, it seems that these are hyperparameters. We have decided to learn them in the training process.
        self.create_embedding(1, emb_type="global", name="a")
        self.create_embedding(1, emb_type="global", name="b")
        self.create_embedding(1, emb_type="global", name="g")
        self.create_embedding(1, emb_type="global", name="w")

    def _calc(self, h, r, t, a, b, g, w):
        # The paper does not specify how to get a single value from h+r-t. However, the different discussions in the
        #   paper like in Section 5.3 ("TransE d = h+r−t = 0") suggests that d is exactly TransE's scoring function.
        # This is Eq. (8). The minus sign is because this is a distance like TransE.
        return -a * torch.sin(w * torch.linalg.norm(h + r - t, dim=-1, ord=self.pnorm) + b) + g

    def return_score(self, is_predict=False):
        (head_emb, rel_emb, tail_emb) = self.current_batch

        h = head_emb["e"]
        t = tail_emb["e"]
        r = rel_emb["r"]

        a, b, g, w = self.current_global_embeddings["a"], self.current_global_embeddings["b"], \
            self.current_global_embeddings["g"], self.current_global_embeddings["w"]

        return self._calc(h, r, t, a, b, g, w)
