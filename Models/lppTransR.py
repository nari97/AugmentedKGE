import torch
from Models.Model import Model


class lppTransR(Model):
    """
    Hee-Geun Yoon, Hyun-Je Song, Seong-Bae Park, Se-Young Park: A Translation-Based Knowledge Graph Embedding Preserving
        Logical Property of Relations. HLT-NAACL 2016: 907-916.
    """
    def __init__(self, ent_total, rel_total, dim_e, dim_r, norm=2):
        """
            ent_total (int): Total number of entities
            rel_total (int): Total number of relations
            dim_e (int): Number of dimensions for entity embeddings
            dim_r (int): Number of dimensions for relation embeddings
            norm (int): L1 or L2 norm. Default: 2 (Eq. (8).).
        """
        super(lppTransR, self).__init__(ent_total, rel_total)
        self.dim_e = dim_e
        self.dim_r = dim_r
        self.pnorm = norm

    def get_default_loss(self):
        # Eq. (10).
        return 'margin'

    def get_score_sign(self):
        # It is a distance (norm).
        return -1

    def initialize_model(self):
        self.create_embedding(self.dim_e, emb_type="entity", name="e")
        self.create_embedding(self.dim_r, emb_type="relation", name="r")
        # We register two matrices.
        self.create_embedding((self.dim_e, self.dim_r), emb_type="relation", name="mrh")
        self.create_embedding((self.dim_e, self.dim_r), emb_type="relation", name="mrt")

        self.register_scale_constraint(emb_type="entity", name="e")
        self.register_scale_constraint(emb_type="relation", name="r")

    def _calc(self, h, r, mrh, mrt, t, is_predict):
        def transfer(e, mr):
            batch_size = e.shape[0]
            return torch.matmul(e.view(batch_size, 1, -1), mr).view(batch_size, self.dim_r)
        hr, tr = transfer(h, mrh), transfer(t, mrt)
        if not is_predict:
            self.onthefly_constraints.append(self.scale_constraint(hr))
            self.onthefly_constraints.append(self.scale_constraint(tr))
        return torch.linalg.norm(hr + r - tr, dim=-1, ord=self.pnorm)

    def return_score(self, is_predict=False):
        (head_emb, rel_emb, tail_emb) = self.current_batch

        h = head_emb["e"]
        t = tail_emb["e"]
        r, mrh, mrt = rel_emb["r"], rel_emb["mrh"], rel_emb["mrt"]

        return self._calc(h, r, mrh, mrt, t, is_predict)
