import torch
from Models.Model import Model
from Utils import PoincareUtils


# Check this: https://github.com/ibalazevic/multirelational-poincare/blob/master/model.py
class MuRP(Model):

    def __init__(self, ent_total, rel_total, dim, c=1):
        super(MuRP, self).__init__(ent_total, rel_total)
        self.dim = dim
        self.c = c

    def get_default_loss(self):
        return 'bce'

    def initialize_model(self):
        self.create_embedding(self.dim, emb_type="entity", name="e")
        self.create_embedding(1, emb_type="entity", name="bh")
        self.create_embedding(1, emb_type="entity", name="bt")
        self.create_embedding(self.dim, emb_type="relation", name="r")
        self.create_embedding(self.dim, emb_type="relation", name="R")

        # This is mandatory so tanh and atanh functions are not applied over a number greater than one.
        self.register_scale_constraint(emb_type="entity", name="e", p=2)
        self.register_scale_constraint(emb_type="relation", name="r", p=2)

    def _calc_train(self, h, bh, r, R, t, bt):
        return -torch.linalg.norm(R*h - t+r, dim=-1, ord=2) + bh.flatten() + bt.flatten()

    def _calc_predict(self, h, bh, r, R, t, bt):
        # Map h and t from Euclidean to Poincare.
        h, t = PoincareUtils.exp_map(h, self.c), PoincareUtils.exp_map(t, self.c)
        # Relation-adjusted head.
        rah = PoincareUtils.exp_map(R * PoincareUtils.log_map(h, self.c), self.c)
        # Relation-adjusted tail.
        rat = PoincareUtils.mobius_addition(t, r, self.c)
        return torch.sigmoid(-PoincareUtils.geodesic_dist(rah, rat, self.c) ** 2 + bh.flatten() + bt.flatten())

    def return_score(self, is_predict=False):
        (head_emb, rel_emb, tail_emb) = self.current_batch

        h = head_emb["e"]
        bh = head_emb["bh"]
        t = tail_emb["e"]
        bt = tail_emb["bt"]
        r, R = rel_emb["r"], rel_emb["R"]

        # This requires a special SGD: https://github.com/ibalazevic/multirelational-poincare/blob/master/rsgd.py
        # Instead, we will optimize the tangent space, then, we map them back to the Poincare ball.
        # Check A.4 in https://arxiv.org/pdf/2005.00545.pdf

        if not is_predict:
            return self._calc_train(h, bh, r, R, t, bt)
        else:
            return self._calc_predict(h, bh, r, R, t, bt)
