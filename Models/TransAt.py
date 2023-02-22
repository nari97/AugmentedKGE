import torch
from Models.Model import Model


class TransAt(Model):
    """
    Wei Qian, Cong Fu, Yu Zhu, Deng Cai, Xiaofei He: Translating Embeddings for Knowledge Graph Completion with Relation
        Attention Mechanism. IJCAI 2018: 4286-4292.
    """
    def __init__(self, ent_total, rel_total, dim):
        """
            dim (int): Number of dimensions for embeddings
        """
        super(TransAt, self).__init__(ent_total, rel_total)
        self.dim = dim

    def get_default_loss(self):
        # Eq. (3). The loss function includes additional terms involving h and t; we are not including them.
        return 'margin'

    def initialize_model(self):
        # See Eq. (2).
        self.create_embedding(self.dim, emb_type="entity", name="e")
        self.create_embedding(self.dim, emb_type="relation", name="r")
        self.create_embedding(self.dim, emb_type="relation", name="rh")
        self.create_embedding(self.dim, emb_type="relation", name="rt")
        # Attention in the paper is either 0 or 1. We use real number between 0 and 1.
        self.create_embedding(self.dim, emb_type="relation", name="a", init_method="uniform", init_params=[0, 1])
        # See below Eq. (3).
        self.register_scale_constraint(emb_type="entity", name="e")
        self.register_scale_constraint(emb_type="relation", name="r")

        # The original paper uses Kmeans to determine "capable" heads and tails for a given relation. We do not use
        #   these; however, the TCLCWA strategy could be used to compute them. Also, these heads and tails and the
        #   attention vector is reset during training. We do not implement these options either.

    def _calc(self, h, r, rh, rt, a, t):
        def proj(x, a):
            # See Section 3.2.
            return x * a

        # Eq. (2). The original paper does not mention how to compute the scores. We assume sum.
        return torch.sum(proj(torch.sigmoid(rh) * h, a) + r - proj(torch.sigmoid(rt) * t, a), dim=-1)

    def return_score(self, is_predict=False):
        (head_emb, rel_emb, tail_emb) = self.current_batch

        h = head_emb["e"]
        t = tail_emb["e"]
        r, rh, rt, a = rel_emb["r"], rel_emb["rh"], rel_emb["rt"], rel_emb["a"]

        return self._calc(h, r, rh, rt, a, t)
