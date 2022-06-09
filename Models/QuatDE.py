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
            self.create_embedding(self.dim, emb_type="entity", name="e_" + component)
            self.create_embedding(self.dim, emb_type="entity", name="p_" + component)
            self.create_embedding(self.dim, emb_type="entity", name="v_" + component)
            self.create_embedding(self.dim, emb_type="relation", name="r_" + component)

            self.register_scale_constraint(emb_type="entity", name="e_" + component)
            self.register_scale_constraint(emb_type="entity", name="p_" + component)
            self.register_scale_constraint(emb_type="entity", name="v_" + component)
            self.register_scale_constraint(emb_type="relation", name="r_" + component)

    def _calc(self, h, ph, vh, r, t, pt, vt):
        # Normalize quaternions.
        nph, nvh = QuaternionUtils.normalize_quaternion(ph), QuaternionUtils.normalize_quaternion(vh)
        npt, nvt = QuaternionUtils.normalize_quaternion(pt), QuaternionUtils.normalize_quaternion(vt)
        nr = QuaternionUtils.normalize_quaternion(r)

        htrans = QuaternionUtils.hamilton_product(h, QuaternionUtils.hamilton_product(nph, nvh))
        ttrans = QuaternionUtils.hamilton_product(t, QuaternionUtils.hamilton_product(npt, nvt))

        return QuaternionUtils.inner_product(QuaternionUtils.hamilton_product(htrans, nr), ttrans)

    def return_score(self, is_predict=False):
        (head_emb, rel_emb, tail_emb) = self.current_batch

        (h_a, h_b, h_c, h_d) = (head_emb["e_a"], head_emb["e_b"], head_emb["e_c"], head_emb["e_d"])
        (t_a, t_b, t_c, t_d) = (tail_emb["e_a"], tail_emb["e_b"], tail_emb["e_c"], tail_emb["e_d"])
        (ph_a, ph_b, ph_c, ph_d) = (head_emb["p_a"], head_emb["p_b"], head_emb["p_c"], head_emb["p_d"])
        (pt_a, pt_b, pt_c, pt_d) = (tail_emb["p_a"], tail_emb["p_b"], tail_emb["p_c"], tail_emb["p_d"])
        (vh_a, vh_b, vh_c, vh_d) = (head_emb["v_a"], head_emb["v_b"], head_emb["v_c"], head_emb["v_d"])
        (vt_a, vt_b, vt_c, vt_d) = (tail_emb["v_a"], tail_emb["v_b"], tail_emb["v_c"], tail_emb["v_d"])
        (r_a, r_b, r_c, r_d) = (rel_emb["r_a"], rel_emb["r_b"], rel_emb["r_c"], rel_emb["r_d"])

        return self._calc((h_a, h_b, h_c, h_d), (ph_a, ph_b, ph_c, ph_d),(vh_a, vh_b, vh_c, vh_d), (r_a, r_b, r_c, r_d),
                          (t_a, t_b, t_c, t_d), (pt_a, pt_b, pt_c, pt_d), (vt_a, vt_b, vt_c, vt_d))
