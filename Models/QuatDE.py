from Models.Model import Model
from Utils import QuaternionUtils


class QuatDE(Model):
    """
    Haipeng Gao, Kun Yang, Yuxue Yang, Rufai Yusuf Zakari, Jim Wilson Owusu, Ke Qin: QuatDE: Dynamic Quaternion
        Embedding for Knowledge Graph Completion. CoRR abs/2105.09002 (2021).
    """
    def __init__(self, ent_total, rel_total, dim):
        """
            dim (int): Number of dimensions for embeddings
        """
        super(QuatDE, self).__init__(ent_total, rel_total)
        self.dim = dim

    def get_default_loss(self):
        # Eq. (13).
        return 'soft'

    def get_score_sign(self):
        # 1 for positives and -1 for negatives, so assuming similarity.
        return 1

    def initialize_model(self):
        for component in ['a', 'b', 'c', 'd']:
            # Regular embeddings like in QuatE.
            self.create_embedding(self.dim, emb_type="entity", name="e_" + component, reg=True)
            self.create_embedding(self.dim, emb_type="entity", name="p_" + component, reg=True)
            # Transfer embeddings.
            # There is a typo in the paper, v is for relations, not entities (above Eq. (11)).
            self.create_embedding(self.dim, emb_type="relation", name="v_" + component, reg=True)
            self.create_embedding(self.dim, emb_type="relation", name="r_" + component, reg=True)

            # All are regularized. From the paper: "the parameters w [in Eq. (13)] for L2 norm include the embedding
            #   vectors and transfer vectors..." See also:
            #   https://github.com/hopkin-ghp/QuatDE/blob/master/models/QuatDE.py#L100

    def _calc(self, h, ph, r, vr, t, pt):
        # Eq. (12).
        # Normalize quaternions.
        nph, npt = QuaternionUtils.normalize_quaternion(ph), QuaternionUtils.normalize_quaternion(pt)
        nr, nvr = QuaternionUtils.normalize_quaternion(r), QuaternionUtils.normalize_quaternion(vr)
        # Hamilton products.
        htrans = QuaternionUtils.hamilton_product(h, QuaternionUtils.hamilton_product(nph, nvr))
        ttrans = QuaternionUtils.hamilton_product(t, QuaternionUtils.hamilton_product(npt, nvr))
        # Final result.
        return QuaternionUtils.inner_product(QuaternionUtils.hamilton_product(htrans, nr), ttrans)

    def return_score(self, is_predict=False):
        (head_emb, rel_emb, tail_emb) = self.current_batch

        h = (head_emb["e_a"], head_emb["e_b"], head_emb["e_c"], head_emb["e_d"])
        t = (tail_emb["e_a"], tail_emb["e_b"], tail_emb["e_c"], tail_emb["e_d"])
        ph = (head_emb["p_a"], head_emb["p_b"], head_emb["p_c"], head_emb["p_d"])
        pt = (tail_emb["p_a"], tail_emb["p_b"], tail_emb["p_c"], tail_emb["p_d"])
        r = (rel_emb["r_a"], rel_emb["r_b"], rel_emb["r_c"], rel_emb["r_d"])
        v = (rel_emb["v_a"], rel_emb["v_b"], rel_emb["v_c"], rel_emb["v_d"])

        return self._calc(h, ph, r, v, t, pt)
