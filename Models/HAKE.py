import math
import torch
from Models.Model import Model


class HAKE(Model):
    """
    Zhanqiu Zhang, Jianyu Cai, Yongdong Zhang, Jie Wang: Learning Hierarchy-Aware Knowledge Graph Embeddings for Link
        Prediction. AAAI 2020: 3065-3072.
    """
    def __init__(self, ent_total, rel_total, dim):
        """
            dim (int): Number of dimensions for embeddings
        """
        super(HAKE, self).__init__(ent_total, rel_total)
        self.dim = dim

    def get_default_loss(self):
        # See Loss Function paragraph before Section 4.
        return 'soft_margin'

    def initialize_model(self):
        # Section 3.
        self.create_embedding(self.dim, emb_type="entity", name="em")
        # All phases must be between 0 and 2*pi (see Table 1).
        self.create_embedding(self.dim, emb_type="entity", name="ep",
                              init_method="uniform", init_params=[0, 2 * math.pi],
                              norm_method="rescaling", norm_params={"a": 0, "b": 2 * math.pi})
        # rm is always positive (see Table 1 and Section 3).
        self.create_embedding(self.dim, emb_type="relation", name="rm",
                              norm_method="rescaling", norm_params={"a": 0})
        # Every element in rmprime must be between 0 and 1. From the paper: "0 < r_m' < 1."
        self.create_embedding(self.dim, emb_type="relation", name="rmprime",
                              norm_method="rescaling", norm_params={"a": 0, "b": 1})
        # All phases must be between 0 and 2*pi (see Table 1).
        self.create_embedding(self.dim, emb_type="relation", name="rp",
                              init_method="uniform", init_params=[0, 2 * math.pi],
                              norm_method="rescaling", norm_params={"a": 0, "b": 2 * math.pi})
        # Section 4.
        self.create_embedding(1, emb_type="global", name="lambda1")
        self.create_embedding(1, emb_type="global", name="lambda2")

    def _calc(self, hm, hp, rm, rmprime, rp, tm, tp, l1, l2):
        # This is d'_{r,m}(h, t) in the paper.
        modulus_scores = torch.linalg.norm(hm*((1-rmprime)/(rm+rmprime))-tm, dim=-1, ord=2)
        # This is d_{r,p}(h, t) in the paper.
        phase_scores = torch.linalg.norm(torch.sin((hp+rp-tp)/2), dim=-1, ord=1)
        # See Training Protocol in Section 4.
        return - l1 * modulus_scores - l2 * phase_scores

    def return_score(self, is_predict=False):
        (head_emb, rel_emb, tail_emb) = self.current_batch

        hm, hp = head_emb["em"], head_emb["ep"]
        tm, tp = tail_emb["em"], tail_emb["ep"]
        rm, rmprime, rp = rel_emb["rm"], rel_emb["rmprime"], rel_emb["rp"]

        l1 = self.current_global_embeddings["lambda1"].item()
        l2 = self.current_global_embeddings["lambda2"].item()

        return self._calc(hm, hp, rm, rmprime, rp, tm, tp, l1, l2)
