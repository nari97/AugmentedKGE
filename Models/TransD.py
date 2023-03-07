import torch
from Models.Model import Model


class TransD(Model):
    """
    Guoliang Ji, Shizhu He, Liheng Xu, Kang Liu, Jun Zhao: Knowledge Graph Embedding via Dynamic Mapping Matrix. ACL (1)
        2015: 687-696.

    TransD :cite:`ji2015knowledge` is a translation-based embedding approach that introduces the concept that entity and relation embeddings are no longer represented in the same space. Entity embeddings are represented in space :math:`\mathbb{R}^{k}` and relation embeddings are represented in space :math:`\mathbb{R}^{d}` where :math:`k \geq d`.TransD also introduces additional embeddings :math:`\mathbf{w_{h}}, \mathbf{w_{t}} \in \mathbb{R}^{k}` and :math:`\mathbf{w_{r} \in \mathbb{R}^{d}}`. I is the identity matrix.
    The scoring function for TransD is defined as

    :math:`f_{r}(h,t) = -||h_{\\bot} + \mathbf{r} - t_{\\bot}||`

    :math:`h_{\\bot} = (\mathbf{w_{r}}\mathbf{w_{h}^{T}} + I^{d \\times k})\,\mathbf{h}`

    :math:`t_{\\bot} = (\mathbf{w_{r}}\mathbf{w_{t}^{T}} + I^{d \\times k})\,\mathbf{t}`

    TransD imposes contraints like :math:`||\mathbf{h}||_{2} \leq 1, ||\mathbf{t}||_{2} \leq 1, ||\mathbf{r}||_{2} \leq 1, ||h_{\\bot}||_{2} \leq 1` and :math:`||t_{\\bot}||_{2} \leq 1`
    """

    def __init__(self, ent_total, rel_total, dim_e, dim_r, norm=2):
        """
        Args:
            ent_total (int): Total number of entities
            rel_total (int): Total number of relations
            dim_e (int): Number of dimensions for entity embeddings
            dim_r (int): Number of dimensions for relation embeddings
            norm (int): L1 or L2 norm. Default: 2 (see Eq. (14)).
        """
        super(TransD, self).__init__(ent_total, rel_total)
        self.dim_e = dim_e
        self.dim_r = dim_r
        self.pnorm = norm

    def get_default_loss(self):
        # Eq. (15).
        return 'margin'

    def get_score_sign(self):
        # It is a distance (norm).
        return -1

    def initialize_model(self):
        # Section 3.2.
        self.create_embedding(self.dim_e, emb_type="entity", name="e")
        self.create_embedding(self.dim_e, emb_type="entity", name="ep")
        self.create_embedding(self.dim_r, emb_type="relation", name="r")
        self.create_embedding(self.dim_r, emb_type="relation", name="rp")
        # See below Eq. (14).
        self.register_scale_constraint(emb_type="entity", name="e")
        self.register_scale_constraint(emb_type="relation", name="r")

    # This method computes each transfer (see Eqs. (11) and (12)).
    def get_et(self, e, ep, rp):
        # Identity matrix.
        i = torch.eye(self.dim_r, self.dim_e, device=e.device)
        batch_size = ep.shape[0]

        # ep changed into a row matrix, and rp changed into a column matrix.
        m = torch.matmul(rp.view(batch_size, -1, 1), ep.view(batch_size, 1, -1))

        # add i to every result in the batch size, multiply by vector and put it back to regular shape.
        return torch.matmul(m + i, e.view(batch_size, -1, 1)).view(batch_size, self.dim_r)

    def _calc(self, h, hp, r, rp, t, tp, is_predict):
        # Get transfers.
        ht = self.get_et(h, hp, rp)
        tt = self.get_et(t, tp, rp)
        # The transfers are also scaled (see below Eq. (14)). Only during training.
        if not is_predict:
            self.onthefly_constraints.append(Model.scale_constraint(ht))
            self.onthefly_constraints.append(Model.scale_constraint(tt))
        # Eq. (14).
        return torch.pow(torch.linalg.norm(ht + r - tt, dim=-1, ord=self.pnorm), 2)

    def return_score(self, is_predict=False):
        (head_emb, rel_emb, tail_emb) = self.current_batch

        h, hp = head_emb["e"], head_emb["ep"]
        t, tp = tail_emb["e"], tail_emb["ep"]
        r, rp = rel_emb["r"], rel_emb["rp"]

        return self._calc(h, hp, r, rp, t, tp, is_predict)
