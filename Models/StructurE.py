import torch
from Models.Model import Model


class StructurE(Model):
    """
    Qianjin Zhang, Ronggui Wang, Juan Yang, Lixia Xue: Structural context-based knowledge graph embedding for link
        prediction. Neurocomputing 470: 109-120 (2022).
    """
    def __init__(self, ent_total, rel_total, dim, norm=2):
        """
            dim (int): Number of dimensions for embeddings
            norm (int): L1 or L2 norm. Default: TODO
        """
        super(StructurE, self).__init__(ent_total, rel_total)
        self.dim = dim
        self.pnorm = norm

    def get_default_loss(self):
        # Eq. (8).
        return 'soft_margin'

    def initialize_model(self):
        # See Table 1.
        self.create_embedding(self.dim, emb_type="entity", name="e")
        self.create_embedding(self.dim, emb_type="entity", name="ec")
        self.create_embedding(self.dim, emb_type="relation", name="r")
        self.create_embedding(self.dim, emb_type="relation", name="rh")
        self.create_embedding(self.dim, emb_type="relation", name="rt")
        # See Section 4.2.2. These apply to each type of distance. In the paper, these are hyperpameters, we include
        #   them as parameters.
        self.create_embedding(1, emb_type="global", name="lr")
        self.create_embedding(1, emb_type="global", name="le")

    def _calc(self, h, hc, r, rh, rt, t, tc, lr, le):
        # This is the interaction for edge structure-context. See Eqs. (1) and (4).
        def interaction_edge_structure_context(e, ec, r, rc):
            return (ec * r + e) * rc
        # Eq. (3). It seems there is a conflict between these equations and Table 1. We use the equations.
        rstruct = torch.linalg.norm(interaction_edge_structure_context(h, hc, r, rh) + r - t, dim=-1 , ord=self.pnorm)
        # Eq. (6).
        estruct = torch.linalg.norm(h + r - interaction_edge_structure_context(t, tc, -r, rt), dim=-1, ord=self.pnorm)
        # Eq. (7). See Section 4.2.2 for lr and le.
        return -lr * rstruct - le * estruct

    def return_score(self, is_predict=False):
        (head_emb, rel_emb, tail_emb) = self.current_batch

        h, hc = head_emb["e"], head_emb["ec"]
        t, tc = tail_emb["e"], tail_emb["ec"]
        rh, rt, r = rel_emb["rh"], rel_emb["rt"], rel_emb["r"]
        lr, le = self.current_global_embeddings["lr"], self.current_global_embeddings["le"]

        return self._calc(h, hc, r, rh, rt, t, tc, lr, le)
