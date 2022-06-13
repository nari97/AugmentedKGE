import torch
from Models.Model import Model
from Utils import QuaternionUtils


class QuatDE(Model):
    def __init__(self, ent_total, rel_total, dim):
        super(QuatDE, self).__init__(ent_total, rel_total)
        self.dim = dim

    def get_default_loss(self):
        return 'soft'

    def initialize_model(self):
        for component in ['a', 'b', 'c', 'd']:
            # All are regularized: https://github.com/hopkin-ghp/QuatDE/blob/master/models/QuatDE.py#L100
            self.create_embedding(self.dim, emb_type="entity", name="e_" + component, reg=True)
            self.create_embedding(self.dim, emb_type="entity", name="p_" + component, reg=True)
            # There is a typo in the paper, v is for relations, not entities.
            self.create_embedding(self.dim, emb_type="relation", name="v_" + component, reg=True)
            self.create_embedding(self.dim, emb_type="relation", name="r_" + component, reg=True)

    def _calc(self, h, ph, r, vr, t, pt, is_predict=False):
        # Normalize quaternions.
        nph, npt = QuaternionUtils.normalize_quaternion(ph), QuaternionUtils.normalize_quaternion(pt)
        nr, nvr = QuaternionUtils.normalize_quaternion(r), QuaternionUtils.normalize_quaternion(vr)

        htrans = QuaternionUtils.hamilton_product(h, QuaternionUtils.hamilton_product(nph, nvr))
        ttrans = QuaternionUtils.hamilton_product(t, QuaternionUtils.hamilton_product(npt, nvr))

        if not is_predict:
            for x in htrans:
                self.register_onthefly_regularization(x)
            for x in ttrans:
                self.register_onthefly_regularization(x)

        return QuaternionUtils.inner_product(QuaternionUtils.hamilton_product(htrans, nr), ttrans)

    def return_score(self, is_predict=False):
        (head_emb, rel_emb, tail_emb) = self.current_batch

        (h_a, h_b, h_c, h_d) = (head_emb["e_a"], head_emb["e_b"], head_emb["e_c"], head_emb["e_d"])
        (t_a, t_b, t_c, t_d) = (tail_emb["e_a"], tail_emb["e_b"], tail_emb["e_c"], tail_emb["e_d"])
        (ph_a, ph_b, ph_c, ph_d) = (head_emb["p_a"], head_emb["p_b"], head_emb["p_c"], head_emb["p_d"])
        (pt_a, pt_b, pt_c, pt_d) = (tail_emb["p_a"], tail_emb["p_b"], tail_emb["p_c"], tail_emb["p_d"])
        (r_a, r_b, r_c, r_d) = (rel_emb["r_a"], rel_emb["r_b"], rel_emb["r_c"], rel_emb["r_d"])
        (v_a, v_b, v_c, v_d) = (rel_emb["v_a"], rel_emb["v_b"], rel_emb["v_c"], rel_emb["v_d"])

        return self._calc((h_a, h_b, h_c, h_d), (ph_a, ph_b, ph_c, ph_d), (r_a, r_b, r_c, r_d), (v_a, v_b, v_c, v_d),
                          (t_a, t_b, t_c, t_d), (pt_a, pt_b, pt_c, pt_d), is_predict=is_predict)
