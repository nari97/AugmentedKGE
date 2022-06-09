import torch
from Models.Model import Model


class TransH(Model):
    """
    TransH :cite:`wang2014knowledge` allows entities to have different representations for different relations by creating an additional embedding :math:`\mathbf{w_{r}} \in \mathbb{R}^{d}`.
    This is done by projecting entities onto a hyperplane specific to the relation r and with normal vector :math:`\mathbf{w_{r}}`.

    :math:`f_r(h,t) = -||h_{\\bot} + \\mathbf{r} -t_{\\bot}||_{2}^{2}`

    :math:`h_{\\bot} = \\mathbf{h} - \\mathbf{w_{r}^{T}}\\mathbf{h} \\mathbf{w_{r}}`

    :math:`t_{\\bot} = \\mathbf{t} - \\mathbf{w_{r}^{T}} \\mathbf{t} \\mathbf{w_{r}}`


    TransH imposes additional constraints :math:`||\mathbf{h}||_{2} \leq 1`, :math:`||\mathbf{t}||_{2} \leq 1` and :math:`||\mathbf{w_{r}}|| = 1`.

    """

    def __init__(self, ent_total, rel_total, dim, norm=2):
        """
        Args:
            ent_total (int): Total number of entities
            rel_total (int): Total number of relations
            dim (int): Number of dimensions for embeddings
        """
        super(TransH, self).__init__(ent_total, rel_total)
        self.dim = dim
        self.pnorm = norm

    def get_default_loss(self):
        return 'margin'

    def initialize_model(self):
        self.create_embedding(self.dim, emb_type="entity", name="e")
        self.create_embedding(self.dim, emb_type="relation", name="r")
        self.create_embedding(self.dim, emb_type="relation", name="w_r", norm_method="norm")

        self.register_scale_constraint(emb_type="entity", name="e")
        self.register_custom_constraint(self.orthogonal_constraint)

    def orthogonal_constraint(self, head_emb, rel_emb, tail_emb, eps=1e-5):
        r, w_r = rel_emb["r"], rel_emb["w_r"]
        constraint = torch.pow(torch.sum(w_r * r, dim=-1), 2)/torch.pow(torch.linalg.norm(r, dim=-1, ord=2), 2) - eps**2
        return torch.maximum(constraint, torch.zeros_like(constraint))

    def _calc(self, h, r, t, w_r):
        ht = h - torch.sum(w_r * h, dim=-1, keepdim=True) * w_r
        tt = t - torch.sum(w_r * t, dim=-1, keepdim=True) * w_r
        return -torch.pow(torch.linalg.norm(ht + r - tt, dim=-1, ord=self.pnorm), 2)

    def return_score(self, is_predict=False):
        (head_emb, rel_emb, tail_emb) = self.current_batch

        h = head_emb["e"]
        t = tail_emb["e"]
        r, w_r = rel_emb["r"], rel_emb["w_r"]

        return self._calc(h, r, t, w_r)
