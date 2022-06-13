import torch
from Models.Model import Model


class ReflectE(Model):
    def __init__(self, ent_total, rel_total, dim, norm=2):
        super(ReflectE, self).__init__(ent_total, rel_total)
        self.dim = dim
        self.pnorm = norm

    def get_default_loss(self):
        return 'soft_margin'

    def initialize_model(self):
        self.create_embedding(self.dim, emb_type="entity", name="e")
        self.create_embedding(self.dim, emb_type="entity", name="ec")
        self.create_embedding(self.dim, emb_type="relation", name="rh")
        self.create_embedding(self.dim, emb_type="relation", name="rt")
        self.create_embedding(self.dim, emb_type="relation", name="rp")

    def get_et(self, e, r):
        # Identity matrix.
        i = torch.eye(self.dim, self.dim, device=e.device)
        batch_size = e.shape[0]
        er = e * r
        return i - 2*torch.bmm(er.view(batch_size, -1, 1), er.view(batch_size, 1, -1))

    def _calc(self, h, hc, rh, rt, rp, t, tc, is_predict):
        batch_size = h.shape[0]

        hcrh = hc * rh
        tcrt = tc * rt

        # Check Eqs. 6 and 7; note that these override Eqs. 2 and 3.
        if not is_predict:
            self.onthefly_constraints.append(self.scale_constraint(hcrh))
            self.onthefly_constraints.append(self.scale_constraint(tcrt))

        hpredict = torch.linalg.norm(
            torch.bmm(self.get_et(hc, rh), t.view(batch_size, -1, 1)).view(batch_size, -1) -
                                     (hcrh + rp), dim=-1 , ord=self.pnorm)
        tpredict = torch.linalg.norm(
            torch.bmm(self.get_et(tc, rt), (h + rp).view(batch_size, -1, 1)).view(batch_size, -1) -
                                     tcrt, dim=-1, ord=self.pnorm)

        # The original paper differentiates between predicting head or tail; we take the minimum between the two.
        return -torch.minimum(hpredict, tpredict)

    def return_score(self, is_predict=False):
        (head_emb, rel_emb, tail_emb) = self.current_batch

        h, hc = head_emb["e"], head_emb["ec"]
        t, tc = tail_emb["e"], tail_emb["ec"]
        rh, rt, rp = rel_emb["rh"], rel_emb["rt"], rel_emb["rp"]

        return self._calc(h, hc, rh, rt, rp, t, tc, is_predict)
