import torch
from Models.MuRE import MuRE
from Utils import PoincareUtils


# Check this: https://github.com/ibalazevic/multirelational-poincare/blob/master/model.py
class MuRP(MuRE):

    def __init__(self, ent_total, rel_total, dim, c=1):
        super(MuRP, self).__init__(ent_total, rel_total, dim)
        self.c = c

    # This requires a special SGD: https://github.com/ibalazevic/multirelational-poincare/blob/master/rsgd.py
    # Instead, we will optimize the tangent space, then, we map them back to the Poincare ball.
    # Check A.4 in https://arxiv.org/pdf/2005.00545.pdf
    def _calc_predict(self, h, bh, r, R, t, bt):
        # Map h and t from Euclidean to Poincare.
        h, t = PoincareUtils.exp_map(h, self.c), PoincareUtils.exp_map(t, self.c)
        # Relation-adjusted head.
        rah = PoincareUtils.exp_map(R * PoincareUtils.log_map(h, self.c), self.c)
        # Relation-adjusted tail.
        rat = PoincareUtils.mobius_addition(t, r, self.c)
        return torch.sigmoid(-PoincareUtils.geodesic_dist(rah, rat, self.c) ** 2 + bh.flatten() + bt.flatten())
