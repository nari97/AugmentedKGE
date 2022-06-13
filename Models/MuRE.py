import math
import torch
from Models.Model import Model

# Check this: https://github.com/ibalazevic/multirelational-poincare/blob/master/model.py
class MuRE(Model):

    def __init__(self, ent_total, rel_total, dim):
        super(MuRE, self).__init__(ent_total, rel_total)
        self.dim = dim

    def get_default_loss(self):
        return 'bce'

    def initialize_model(self):
        self.create_embedding(self.dim, emb_type="entity", name="e")
        # Two parameters (unclear in the paper): https://github.com/ibalazevic/multirelational-poincare/blob/master/model.py#L15
        self.create_embedding(1, emb_type="entity", name="bh")
        self.create_embedding(1, emb_type="entity", name="bt")
        self.create_embedding(self.dim, emb_type="relation", name="r")
        self.create_embedding(self.dim, emb_type="relation", name="R")

        # There are no scale constraints in the paper; however, they are in the source code:
        # https://github.com/ibalazevic/multirelational-poincare/blob/master/model.py#L26
        # https://github.com/ibalazevic/multirelational-poincare/blob/master/model.py#L36
        # When we include them, the loss increases, so we remove them.
        #self.register_scale_constraint(emb_type="entity", name="e")
        #self.register_scale_constraint(emb_type="relation", name="r")
        #self.register_scale_constraint(emb_type="relation", name="R")

    # In this case, the predict and train are the same. They are not in MuRP.
    def _calc_predict(self, h, bh, r, R, t, bt):
        return self._calc_train(h, bh, r, R, t, bt, is_predict=True)

    def _calc_train(self, h, bh, r, R, t, bt, is_predict=False):
        rh = R*h
        tr = t+r

        #if not is_predict:
        #    self.onthefly_constraints.append(self.scale_constraint(rh))
        #    self.onthefly_constraints.append(self.scale_constraint(tr))

        return -torch.linalg.norm(rh - tr, dim=-1, ord=2)**2 + bh.flatten() + bt.flatten()

    def return_score(self, is_predict=False):
        (head_emb, rel_emb, tail_emb) = self.current_batch

        h, bh = head_emb["e"], head_emb["bh"]
        t, bt = tail_emb["e"], tail_emb["bt"]
        r, R = rel_emb["r"], rel_emb["R"]

        if not is_predict:
            return self._calc_train(h, bh, r, R, t, bt)
        else:
            return self._calc_predict(h, bh, r, R, t, bt)
