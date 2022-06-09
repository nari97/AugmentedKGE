import torch
from Models.Model import Model


class TransDR(Model):
    def __init__(self, ent_total, rel_total, dim_e, dim_r, norm=2):
        super(TransDR, self).__init__(ent_total, rel_total)
        self.dim_e = dim_e
        self.dim_r = dim_r
        self.pnorm = norm

    def get_default_loss(self):
        return 'margin'

    def initialize_model(self):
        self.create_embedding(self.dim_e, emb_type="entity", name="e")
        self.create_embedding(self.dim_e, emb_type="entity", name="ep")
        self.create_embedding(self.dim_r, emb_type="relation", name="r")
        self.create_embedding(self.dim_r, emb_type="relation", name="rp")

        self.register_scale_constraint(emb_type="entity", name="e")
        self.register_scale_constraint(emb_type="relation", name="r")

    def get_et(self, e, ep, rp):
        # Identity matrix.
        i = torch.eye(self.dim_r, self.dim_e, device=e.device)
        batch_size = ep.shape[0]

        # ep changed into a row matrix, and rp changed into a column matrix.
        m = torch.matmul(rp.view(batch_size, -1, 1), ep.view(batch_size, 1, -1))

        # add i to every result in the batch size, multiply by vector and put it back to regular shape.
        return torch.matmul(m + i, e.view(batch_size, -1, 1)).view(batch_size, self.dim_r)

    def _calc(self, h, hp, r, rp, t, tp, is_predict):
        ht = self.get_et(h, hp, rp)
        tt = self.get_et(t, tp, rp)

        result = ht + r - tt
        wr = 1 / torch.std(result, dim=1)

        if not is_predict:
            self.onthefly_constraints.append(self.scale_constraint(ht))
            self.onthefly_constraints.append(self.scale_constraint(tt))
            # The paper says ||wr||=1, but they implement it as ||wr||>=1.
            self.onthefly_constraints.append(self.scale_constraint(wr, ctype='ge'))

        return -torch.pow(torch.linalg.norm(wr.view(-1, 1) * result, ord=self.pnorm, dim=-1), 2)

    def return_score(self, is_predict=False):
        (head_emb, rel_emb, tail_emb) = self.current_batch

        h, hp = head_emb["e"], head_emb["ep"]
        t, tp = tail_emb["e"], tail_emb["ep"]
        r, rp = rel_emb["r"], rel_emb["rp"]

        return self._calc(h, hp, r, rp, t, tp, is_predict)
