import torch
from Models.Model import Model


class StructurE(Model):
    def __init__(self, ent_total, rel_total, dim, norm=2):
        super(StructurE, self).__init__(ent_total, rel_total)
        self.dim = dim
        self.pnorm = norm

    def get_default_loss(self):
        return 'soft_margin'

    def initialize_model(self):
        self.create_embedding(self.dim, emb_type="entity", name="e")
        self.create_embedding(self.dim, emb_type="entity", name="ec")
        self.create_embedding(self.dim, emb_type="relation", name="r")
        self.create_embedding(self.dim, emb_type="relation", name="rh")
        self.create_embedding(self.dim, emb_type="relation", name="rt")
        self.create_embedding(1, emb_type="global", name="lr")
        self.create_embedding(1, emb_type="global", name="le")

    def get_score_func(self, e, ec, r, rc):
        return (ec * r + e) * rc

    def _calc(self, h, hc, r, rh, rt, t, tc, lr, le):
        rel_struct = torch.linalg.norm(h + r - self.get_score_func(t, tc, -r, rt), dim=-1 , ord=self.pnorm)
        edge_struct = torch.linalg.norm(t - r - self.get_score_func(h, hc, r, rh), dim=-1, ord=self.pnorm)
        return -lr * rel_struct - le * edge_struct

    def return_score(self, is_predict=False):
        (head_emb, rel_emb, tail_emb) = self.current_batch

        h, hc = head_emb["e"], head_emb["ec"]
        t, tc = tail_emb["e"], tail_emb["ec"]
        rh, rt, r = rel_emb["rh"], rel_emb["rt"], rel_emb["r"]
        lr, le = self.current_global_embeddings["lr"], self.current_global_embeddings["le"]

        return self._calc(h, hc, r, rh, rt, t, tc, lr, le)
