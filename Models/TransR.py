import torch
from Models.Model import Model


# LinearRE reports it cannot be run even with huge resources.
class TransR(Model):
    """
    Yankai Lin, Zhiyuan Liu, Maosong Sun, Yang Liu, Xuan Zhu: Learning Entity and Relation Embeddings for Knowledge
        Graph Completion. AAAI 2015: 2181-2187.
    CTransR, proposed also in this paper, requires previous training for clustering. We do not implement CTransR.
    """
    def __init__(self, ent_total, rel_total, dim_e, dim_r, norm=2):
        """
            ent_total (int): Total number of entities
            rel_total (int): Total number of relations
            dim_e (int): Number of dimensions for entity embeddings
            dim_r (int): Number of dimensions for relation embeddings
            norm (int): L1 or L2 norm. Default: 2 (Eq. (8).).
        """
        super(TransR, self).__init__(ent_total, rel_total)
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
        # See TransR section.
        self.create_embedding(self.dim_e, emb_type="entity", name="e")
        self.create_embedding(self.dim_r, emb_type="relation", name="r")
        self.create_embedding((self.dim_e, self.dim_r), emb_type="relation", name="mr")
        # See below Eq. (8).
        self.register_scale_constraint(emb_type="entity", name="e")
        self.register_scale_constraint(emb_type="relation", name="r")

    def _calc(self, h, r, mr, t, is_predict):
        # This method computes the transfer.
        def transfer(e):
            # Eq. (7).
            batch_size = e.shape[0]
            # Change e into a row matrix, multiply and get final result as dim_r
            return torch.matmul(e.view(batch_size, 1, -1), mr).view(batch_size, self.dim_r)
        # Transfers.
        hr, tr = transfer(h), transfer(t)
        # See below Eq. (8).
        if not is_predict:
            self.onthefly_constraints.append(self.scale_constraint(hr))
            self.onthefly_constraints.append(self.scale_constraint(tr))
        # Eq. (8).
        return torch.linalg.norm(hr + r - tr, dim=-1, ord=self.pnorm)

    def return_score(self, is_predict=False):
        (head_emb, rel_emb, tail_emb) = self.current_batch

        h = head_emb["e"]
        t = tail_emb["e"]
        r, mr = rel_emb["r"], rel_emb["mr"]

        return self._calc(h, r, mr, t, is_predict)
