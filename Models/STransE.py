import torch
from Models.Model import Model


class STransE(Model):
    """
    Dat Quoc Nguyen, Kairit Sirts, Lizhen Qu, Mark Johnson: STransE: a novel embedding model of entities and
        relationships in knowledge bases. HLT-NAACL 2016: 460-466.
    """
    def __init__(self, ent_total, rel_total, dim, norm=1):
        """
            dim (int): Number of dimensions for embeddings
            norm (int): L1 or L2 norm. Default: 1. From the paper: "...in our experiments we found that the L1 norm gave
                slightly better results..."
        """
        super(STransE, self).__init__(ent_total, rel_total)
        self.dim = dim
        self.pnorm = norm

    def get_default_loss(self):
        # See Section 2.
        return 'margin'

    def get_score_sign(self):
        # It is a distance ("distance-based score function").
        return -1

    def initialize_model(self):
        # See Table 1.
        self.create_embedding(self.dim, emb_type="entity", name="e")
        self.create_embedding(self.dim, emb_type="relation", name="r")
        self.create_embedding((self.dim, self.dim), emb_type="relation", name="wr1")
        self.create_embedding((self.dim, self.dim), emb_type="relation", name="wr2")
        # See at the end of Section 2.
        self.register_scale_constraint(emb_type="entity", name="e")
        self.register_scale_constraint(emb_type="relation", name="r")

    def get_et(self, w, e):
        batch_size = e.shape[0]
        #  multiply by vector and put it back to regular shape.
        return torch.matmul(w, e.view(batch_size, -1, 1)).view(batch_size, self.dim)

    def _calc(self, h, wr1, r, t, wr2, is_predict):
        # These are the transfers.
        ht = self.get_et(wr1, h)
        tt = self.get_et(wr2, t)

        if not is_predict:
            # The transfers are constrained (see at the end of Section 2).
            self.onthefly_constraints.append(self.scale_constraint(ht))
            self.onthefly_constraints.append(self.scale_constraint(tt))

        # See Table 1 and Section 2.
        return torch.linalg.norm(ht + r - tt, dim=-1, ord=self.pnorm)

    def return_score(self, is_predict=False):
        (head_emb, rel_emb, tail_emb) = self.current_batch

        h = head_emb["e"]
        t = tail_emb["e"]
        r, wr1, wr2 = rel_emb["r"], rel_emb["wr1"], rel_emb["wr2"]

        return self._calc(h, wr1, r, t, wr2, is_predict)
