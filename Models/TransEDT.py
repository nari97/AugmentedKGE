import torch
from Models.TransE import TransE


class TransEDT(TransE):
    """
    Liang Chang, Manli Zhu, Tianlong Gu, Chenzhong Bin, Junyan Qian, Ji Zhang: Knowledge Graph Embedding by Dynamic
        Translation. IEEE Access 5: 20898-20907 (2017).
        # TODO: Work on this one!
    """
    def initialize_model(self):
        super().initialize_model()
        self.create_embedding(self.dim, emb_type="entity", name="ehalpha")
        self.create_embedding(self.dim, emb_type="entity", name="etalpha")
        self.create_embedding(self.dim, emb_type="relation", name="ralpha")
        # From the paper: "...the L2-norm of them are values from the set {.1, .2, .3}."
        self.register_scale_constraint(emb_type="entity", name="ehalpha", z=.3)
        self.register_scale_constraint(emb_type="entity", name="etalpha", z=.3)
        self.register_scale_constraint(emb_type="relation", name="ralpha", z=.3)

    def _calc(self, h, halpha, r, ralpha, t, talpha):
        return -torch.linalg.norm((h + halpha) + (r + ralpha) - (t + talpha), dim=-1, ord=self.pnorm)

    def return_score(self, is_predict=False):
        (head_emb, rel_emb, tail_emb) = self.current_batch

        h, halpha = head_emb["e"], head_emb["ehalpha"]
        t, talpha = tail_emb["e"], tail_emb["etalpha"]
        r, ralpha = rel_emb["r"], rel_emb["ralpha"]

        return self._calc(h, halpha, r, ralpha, t, talpha)
