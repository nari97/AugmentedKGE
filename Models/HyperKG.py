import math
import torch
from Models.Model import Model
from Utils import PoincareUtils


class HyperKG(Model):
    def __init__(self, ent_total, rel_total, dim, beta=None):
        super(HyperKG, self).__init__(ent_total, rel_total)
        self.dim = dim
        if beta is None:
            self.beta = math.floor(self.dim/2)
        else:
            self.beta = beta
        # Note: we do not include c as we are not learning in the Poincare space.
        self.c = 1

    def get_default_loss(self):
        return 'margin'

    def initialize_model(self):
        self.create_embedding(self.dim, emb_type="entity", name="e")
        self.create_embedding(self.dim, emb_type="relation", name="r")

        # Eq. 8: 1 - ||emb||^2, which is equivalent to ||emb|| >= 1.
        self.register_scale_constraint(emb_type="entity", name="e", ctype='ge')
        self.register_scale_constraint(emb_type="relation", name="r", ctype='ge')

        # Eq. 9: ||emb||<=.5 if entity; ||emb||<=1 if relation.
        self.register_scale_constraint(emb_type="entity", name="e", z=.5)
        self.register_scale_constraint(emb_type="relation", name="r")

    def _calc_train(self, h, r, t):
        # We want to rotate t beta times.
        t_rolled = torch.roll(t, shifts=self.beta, dims=1)

        return -torch.linalg.norm(h + t_rolled - r, dim=-1, ord=2)

    def _calc_predict(self, h, r, t):
        # Map h, r and t from Euclidean to Poincare.
        h, r, t = PoincareUtils.exp_map(h, self.c), PoincareUtils.exp_map(r, self.c), PoincareUtils.exp_map(t, self.c)

        t_rolled = torch.roll(t, shifts=self.beta, dims=1)

        return -PoincareUtils.distance(h + t_rolled, r)

    def return_score(self, is_predict=False):
        (head_emb, rel_emb, tail_emb) = self.current_batch

        h = head_emb["e"]
        t = tail_emb["e"]
        r = rel_emb["r"]

        # This requires a special SGD: https://github.com/ibalazevic/multirelational-poincare/blob/master/rsgd.py
        # Instead, we will optimize the tangent space, then, we map them back to the Poincare ball.
        # Check A.4 in https://arxiv.org/pdf/2005.00545.pdf

        if not is_predict:
            return self._calc_train(h, r, t)
        else:
            return self._calc_predict(h, r, t)
