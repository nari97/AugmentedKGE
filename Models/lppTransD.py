import torch
from Models.Model import Model


class lppTransD(Model):
    """
    Hee-Geun Yoon, Hyun-Je Song, Seong-Bae Park, Se-Young Park: A Translation-Based Knowledge Graph Embedding Preserving
        Logical Property of Relations. HLT-NAACL 2016: 907-916.
    """

    def __init__(self, ent_total, rel_total, dim_e, dim_r, norm=2):
        """
        Args:
            ent_total (int): Total number of entities
            rel_total (int): Total number of relations
            dim_e (int): Number of dimensions for entity embeddings
            dim_r (int): Number of dimensions for relation embeddings
            norm (int): L1 or L2 norm. Default: 2.
        """
        super(lppTransD, self).__init__(ent_total, rel_total)
        self.dim_e = dim_e
        self.dim_r = dim_r
        self.pnorm = norm

    def get_default_loss(self):
        # Eq. (15).
        return 'margin'

    def get_score_sign(self):
        # It is a distance (norm).
        return -1

    def initialize_model(self):
        self.create_embedding(self.dim_e, emb_type="entity", name="e")
        self.create_embedding(self.dim_e, emb_type="entity", name="ep")
        self.create_embedding(self.dim_r, emb_type="relation", name="r")
        # Create two different embeddings.
        self.create_embedding(self.dim_r, emb_type="relation", name="rph")
        self.create_embedding(self.dim_r, emb_type="relation", name="rpt")

        self.register_scale_constraint(emb_type="entity", name="e")
        self.register_scale_constraint(emb_type="relation", name="r")

    def get_et(self, e, ep, rp):
        i = torch.eye(self.dim_r, self.dim_e, device=e.device)
        batch_size = ep.shape[0]
        m = torch.matmul(rp.view(batch_size, -1, 1), ep.view(batch_size, 1, -1))
        return torch.matmul(m + i, e.view(batch_size, -1, 1)).view(batch_size, self.dim_r)

    def _calc(self, h, hp, r, rph, rpt, t, tp, is_predict):
        ht = self.get_et(h, hp, rph)
        tt = self.get_et(t, tp, rpt)
        if not is_predict:
            self.onthefly_constraints.append(self.scale_constraint(ht))
            self.onthefly_constraints.append(self.scale_constraint(tt))
        return torch.pow(torch.linalg.norm(ht + r - tt, dim=-1, ord=self.pnorm), 2)

    def return_score(self, is_predict=False):
        (head_emb, rel_emb, tail_emb) = self.current_batch

        h, hp = head_emb["e"], head_emb["ep"]
        t, tp = tail_emb["e"], tail_emb["ep"]
        r, rph, rpt = rel_emb["r"], rel_emb["rph"], rel_emb["rpt"]

        return self._calc(h, hp, r, rph, rpt, t, tp, is_predict)
