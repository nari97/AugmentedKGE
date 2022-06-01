import torch
from Models.Model import Model


class TransD(Model):
    """
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
        """
        super(TransD, self).__init__(ent_total, rel_total)
        self.dim_e = dim_e
        self.dim_r = dim_r
        self.pnorm = norm

    def get_default_loss(self):
        return 'margin'

    def initialize_model(self):
        self.create_embedding(self.dim_e, emb_type="entity", name="e")
        self.create_embedding(self.dim_e, emb_type="entity", name="ep")
        self.create_embedding(self.dim_r, emb_type="relation", name="r")
        self.create_embedding(self.dim_r, emb_type="relation", name="rp")

        self.register_scale_constraint(emb_type="entity", name="e", p=2)
        self.register_scale_constraint(emb_type="relation", name="r", p=2)
        self.register_custom_constraint(self.h_constraint)
        self.register_custom_constraint(self.t_constraint)

    def h_constraint(self, head_emb, rel_emb, tail_emb):
        h = head_emb["e"]
        hp = head_emb["ep"]
        rp = rel_emb["rp"]
        return self.max_clamp(torch.linalg.norm(self.get_et(h, hp, rp), dim=-1, ord=2), 1)

    def t_constraint(self, head_emb, rel_emb, tail_emb):
        t = tail_emb["e"]
        tp = tail_emb["ep"]
        rp = rel_emb["rp"]
        return self.max_clamp(torch.linalg.norm(self.get_et(t, tp, rp), dim=-1, ord=2), 1)

    def get_et(self, e, ep, rp):
        # Identity matrix.
        i = torch.eye(self.dim_r, self.dim_e, device=e.device)
        batch_size = ep.shape[0]

        # ep changed into a row matrix, and rp changed into a column matrix.
        m = torch.matmul(rp.view(batch_size, -1, 1), ep.view(batch_size, 1, -1))

        # add i to every result in the batch size, multiply by vector and put it back to regular shape.
        return torch.matmul(m + i, e.view(batch_size, -1, 1)).view(batch_size, self.dim_r)

    def _calc(self, h, hp, r, rp, t, tp):
        return -torch.pow(torch.linalg.norm(
            self.get_et(h, hp, rp) + r - self.get_et(t, tp, rp), ord=self.pnorm, dim=-1), 2)

    def return_score(self, is_predict=False):
        (head_emb, rel_emb, tail_emb) = self.current_batch

        h = head_emb["e"]
        hp = head_emb["ep"]
        t = tail_emb["e"]
        tp = tail_emb["ep"]
        r = rel_emb["r"]
        rp = rel_emb["rp"]

        return self._calc(h, hp, r, rp, t, tp)
