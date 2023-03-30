import torch
from Models.Model import Model


class ReflectE(Model):
    """
    Qianjin Zhang, Ronggui Wang, Juan Yang, Lixia Xue: Knowledge graph embedding by reflection transformation. Knowl.
        Based Syst. 238: 107861 (2022).
    """
    def __init__(self, ent_total, rel_total, dim, norm=2, variant="full"):
        """
            dim (int): Number of dimensions for embeddings
            norm (int): L1 or L2 norm. Default: 2 (just because).
            variant can be either s, b, m or full.
        """
        super(ReflectE, self).__init__(ent_total, rel_total)
        self.dim = dim
        self.pnorm = norm
        self.variant = variant

    def get_default_loss(self):
        # Eq. (10).
        return 'soft_margin'

    def get_score_sign(self):
        # It is a distance (norm).
        return -1

    def initialize_model(self):
        self.create_embedding(self.dim, emb_type="entity", name="e")
        self.create_embedding(self.dim, emb_type="entity", name="ec")

        if self.variant == 's' or self.variant == 'b':
            # In these variants, ||rh||_2=1 and ||rt||_2=1. See below Eqs. (2) and (3).
            self.create_embedding(self.dim, emb_type="relation", name="rh", norm_method="norm")
            self.create_embedding(self.dim, emb_type="relation", name="rt", norm_method="norm")
        else:
            # For the other variants, no norms.
            self.create_embedding(self.dim, emb_type="relation", name="rh")
            self.create_embedding(self.dim, emb_type="relation", name="rt")

        # rp is not present in the s variant.
        if self.variant != 's':
            self.create_embedding(self.dim, emb_type="relation", name="rp")

    def _calc(self, h, hc, rh, rt, rp, t, tc, is_predict):
        batch_size = h.shape[0]

        def get_mr(x):
            # This implements I - 2*x*x^T, which is the main component of Eqs. (2), (3), (6) and (7).
            return 1 - 2 * torch.bmm(x.view(batch_size, -1, 1), x.view(batch_size, 1, -1))

        if self.variant == 's' or self.variant == 'b':
            # Eqs. (2) and (3).
            mrh, mrt = get_mr(rh), get_mr(rt)
        else:
            # Eqs. (6) and (7).
            rhrc, rttc = rh * hc, rt * tc
            mrh, mrt = get_mr(rhrc), get_mr(rttc)

        if self.variant == 's':
            # Eq. (2).
            head_preds = torch.linalg.norm(torch.bmm(mrt, t.view(batch_size, -1, 1)).view(batch_size, -1) - hc,
                                           dim=-1 , ord=self.pnorm)
            # Eq. (3).
            tail_preds = torch.linalg.norm(torch.bmm(mrh, h.view(batch_size, -1, 1)).view(batch_size, -1) - tc,
                                           dim=-1, ord=self.pnorm)

        if self.variant == 'b' or self.variant == 'm':
            # Eq. (4).
            head_preds = torch.linalg.norm(torch.bmm(mrt, t.view(batch_size, -1, 1)).view(batch_size, -1) - (hc + rp),
                                           dim=-1 , ord=self.pnorm)
            # Eq. (5).
            tail_preds = torch.linalg.norm(torch.bmm(mrh, (h + rp).view(batch_size, -1, 1)).view(batch_size, -1) - tc,
                                           dim=-1, ord=self.pnorm)

        if self.variant == 'full':
            # Eq. (8).
            head_preds = torch.linalg.norm(torch.bmm(mrt, t.view(batch_size, -1, 1)).view(batch_size, -1) - (hc*rh+rp),
                                           dim=-1 , ord=self.pnorm)
            # Eq. (5).
            tail_preds = torch.linalg.norm(torch.bmm(mrh, (h+rp).view(batch_size, -1, 1)).view(batch_size, -1) - tc*rt,
                                           dim=-1, ord=self.pnorm)

        # Check Eqs. (6) and (7).
        if not is_predict and (self.variant == 'm' or self.variant == 'full'):
            # The original paper says ||_||_2=1; we implement ||_||_2<=1 and ||_||_2>=1
            self.onthefly_constraints.append(self.scale_constraint(rhrc))
            self.onthefly_constraints.append(self.scale_constraint(rttc))
            self.onthefly_constraints.append(self.scale_constraint(rhrc, ctype='ge'))
            self.onthefly_constraints.append(self.scale_constraint(rttc, ctype='ge'))

        # The original paper differentiates between predicting head or tail; we take the minimum between the two.
        return torch.minimum(head_preds, tail_preds)

    def return_score(self, is_predict=False):
        (head_emb, rel_emb, tail_emb) = self.current_batch

        h, hc = head_emb["e"], head_emb["ec"]
        t, tc = tail_emb["e"], tail_emb["ec"]
        rh, rt, rp = rel_emb["rh"], rel_emb["rt"], rel_emb.get("rp", None)

        return self._calc(h, hc, rh, rt, rp, t, tc, is_predict)
