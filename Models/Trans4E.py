import torch
from Models.Model import Model
from Utils import QuaternionUtils


class Trans4E(Model):
    """
    Mojtaba Nayyeri, Gökce Müge Cil, Sahar Vahdati, Francesco Osborne, Mahfuzur Rahman, Simone Angioni, Angelo A.
        Salatino, Diego Reforgiato Recupero, Nadezhda Vassilyeva, Enrico Motta, Jens Lehmann: Trans4E: Link prediction
        on scholarly knowledge graphs. Neurocomputing 461: 530-542 (2021).
    The paper proposes two variants (Trans4EReg1 and Trans4EReg2), but there are not enough details to know what they
        are.
    """
    def __init__(self, ent_total, rel_total, dim, norm=1):
        """
            dim (int): Number of dimensions for embeddings
            norm (int): L1 or L2 norm. Default: 1.
        """
        super(Trans4E, self).__init__(ent_total, rel_total)
        self.dim = dim
        self.pnorm = norm

    def get_default_loss(self):
        # Not specified; using the same as QuatE.
        return 'soft'

    def get_score_sign(self):
        # It is a norm.
        return -1

    def initialize_model(self):
        # See Section 4.1. Quaternions for entities and relations, and another nabla, which seems to be for entities.
        for component in ['a', 'b', 'c', 'd']:
            self.create_embedding(self.dim, emb_type="entity", name="e_" + component)
            self.create_embedding(self.dim, emb_type="entity", name="nabla_" + component)
            self.create_embedding(self.dim, emb_type="relation", name="r_" + component)

    def _calc(self, h, r, nabla, t):
        batch_size = h[0].shape[0]
        # Rotated h using r. See Section 4.1, Step (a).
        hr = QuaternionUtils.hamilton_product(h, r)
        # Adjusted t. See Section 4.1, Step (c).
        nt = QuaternionUtils.hamilton_product(nabla, t)
        # Eq. (9). It is unclear how to apply this norm. Note that it is quaternion norm (denoted by |.|, see Eq. (5)).
        #   We apply norms to each quaternion component and add them.
        scores = torch.zeros(batch_size, dtype=h[0].dtype, device=h[0].device)
        for i in range(4):
            scores += torch.linalg.norm(hr[i] + r[i] - nt[i], dim=-1, ord=self.pnorm)
        return scores

    def return_score(self, is_predict=False):
        (head_emb, rel_emb, tail_emb) = self.current_batch

        h = (head_emb["e_a"], head_emb["e_b"], head_emb["e_c"], head_emb["e_d"])
        t = (tail_emb["e_a"], tail_emb["e_b"], tail_emb["e_c"], tail_emb["e_d"])
        r = (rel_emb["r_a"], rel_emb["r_b"], rel_emb["r_c"], rel_emb["r_d"])
        # The paper seems to imply that nabla depends on heads only.
        nabla = (head_emb["nabla_a"], head_emb["nabla_b"], head_emb["nabla_c"], head_emb["nabla_d"])

        return self._calc(h, r, nabla, t)
