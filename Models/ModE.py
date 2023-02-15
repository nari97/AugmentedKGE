import torch
from Models.Model import Model


class ModE(Model):
    """
    Zhanqiu Zhang, Jianyu Cai, Yongdong Zhang, Jie Wang: Learning Hierarchy-Aware Knowledge Graph Embeddings for Link
        Prediction. AAAI 2020: 3065-3072.
    See Section 4. This is only the modulus part.
    """
    def __init__(self, ent_total, rel_total, dim):
        """
            dim (int): Number of dimensions for embeddings
        """
        super(ModE, self).__init__(ent_total, rel_total)
        self.dim = dim

    def get_default_loss(self):
        # See Loss Function paragraph before Section 4.
        return 'soft_margin'

    def initialize_model(self):
        # Section 3.
        self.create_embedding(self.dim, emb_type="entity", name="em")
        self.create_embedding(self.dim, emb_type="relation", name="rm")
        self.create_embedding(self.dim, emb_type="relation", name="rmprime")
        # Section 4.
        self.create_embedding(1, emb_type="global", name="lambda1")

        # Every element in rmprime must be between 0 and 1. From the paper: "0 < r_m' < 1."
        self.register_scale_constraint(emb_type="relation", name="rmprime")

    def _calc(self, hm, rm, rmprime, tm, l1):
        # This is d'_{r,m}(h, t) in the paper.
        modulus_scores = torch.linalg.norm(hm*((1-rmprime)/(rm+rmprime))-tm, dim=-1, ord=2)
        # See Training Protocol in Section 4.
        return - l1 * modulus_scores

    def return_score(self, is_predict=False):
        (head_emb, rel_emb, tail_emb) = self.current_batch

        hm = head_emb["em"]
        tm = tail_emb["em"]
        rm, rmprime = rel_emb["rm"], rel_emb["rmprime"]

        l1 = self.current_global_embeddings["lambda1"].item()

        return self._calc(hm, rm, rmprime, tm, l1)
