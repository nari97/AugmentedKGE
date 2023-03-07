import torch
from Models.Model import Model


class TransH(Model):
    """
    Zhen Wang, Jianwen Zhang, Jianlin Feng, Zheng Chen: Knowledge Graph Embedding by Translating on Hyperplanes. AAAI
        2014: 1112-1119.

    TransH :cite:`wang2014knowledge` allows entities to have different representations for different relations by creating an additional embedding :math:`\mathbf{w_{r}} \in \mathbb{R}^{d}`.
    This is done by projecting entities onto a hyperplane specific to the relation r and with normal vector :math:`\mathbf{w_{r}}`.

    :math:`f_r(h,t) = -||h_{\\bot} + \\mathbf{r} -t_{\\bot}||_{2}^{2}`

    :math:`h_{\\bot} = \\mathbf{h} - \\mathbf{w_{r}^{T}}\\mathbf{h} \\mathbf{w_{r}}`

    :math:`t_{\\bot} = \\mathbf{t} - \\mathbf{w_{r}^{T}} \\mathbf{t} \\mathbf{w_{r}}`


    TransH imposes additional constraints :math:`||\mathbf{h}||_{2} \leq 1`, :math:`||\mathbf{t}||_{2} \leq 1` and :math:`||\mathbf{w_{r}}|| = 1`.

    """
    def __init__(self, ent_total, rel_total, dim, norm=2):
        """
            dim (int): Number of dimensions for embeddings
            norm (int): L1 or L2 norm. Default: 2 (see Table 1).
        """
        super(TransH, self).__init__(ent_total, rel_total)
        self.dim = dim
        self.pnorm = norm

    def get_default_loss(self):
        # Eq. (4).
        return 'margin'

    def get_score_sign(self):
        # It is a distance (norm).
        return -1

    def initialize_model(self):
        # See Translating on Hyperplanes (TransH) section.
        self.create_embedding(self.dim, emb_type="entity", name="e")
        self.create_embedding(self.dim, emb_type="relation", name="r")
        # See Eq. (3) for constraint.
        self.create_embedding(self.dim, emb_type="relation", name="w_r", norm_method="norm")
        # Eq. (1).
        self.register_scale_constraint(emb_type="entity", name="e")
        # Eq. (2).
        self.register_custom_constraint(TransH.orthogonal_constraint)

    @staticmethod
    def orthogonal_constraint(head_emb, rel_emb, tail_emb, eps=1e-5):
        # Eq. (4).
        r, w_r = rel_emb["r"], rel_emb["w_r"]
        # w_r^T*r = sum (w_r^T)_i*r_i
        mult = torch.sum(w_r * r, dim=-1).view(-1, 1)
        constraint = torch.sum(
            torch.pow(mult, 2)/torch.pow(torch.linalg.norm(r, dim=-1, ord=2).view(-1, 1), 2) - eps**2, dim=-1)
        return torch.maximum(constraint, torch.zeros_like(constraint))

    def _calc(self, h, r, t, w_r):
        def transfer(e):
            # w_r^T*e = sum (w_r^T)_i*e_i
            mult = torch.sum(w_r*e, dim=-1).view(-1, 1)
            return e - mult * w_r
        # Get transfers.
        ht, tt = transfer(h), transfer(t)
        # See score function.
        return torch.pow(torch.linalg.norm(ht + r - tt, dim=-1, ord=self.pnorm), 2)

    def return_score(self, is_predict=False):
        (head_emb, rel_emb, tail_emb) = self.current_batch

        h = head_emb["e"]
        t = tail_emb["e"]
        r, w_r = rel_emb["r"], rel_emb["w_r"]

        return self._calc(h, r, t, w_r)
