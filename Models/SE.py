import torch
from Models.Model import Model


class SE(Model):
    """
    Antoine Bordes, Jason Weston, Ronan Collobert, Yoshua Bengio: Learning Structured Embeddings of Knowledge Bases.
        AAAI 2011.
    """
    def __init__(self, ent_total, rel_total, dim, norm=1):
        """
            dim (int): Number of dimensions for embeddings
            norm (int): L1 or L2 norm. Default: 1 ("...due to the simplicity of gradient-based learning...").
        """
        super(SE, self).__init__(ent_total, rel_total)
        self.dim = dim
        self.pnorm = norm

    def get_default_loss(self):
        # Step 3 in the training subsection.
        return 'margin'

    def initialize_model(self):
        # See below Eq. (3). The norm of e is mentioned in the training subsection.
        self.create_embedding(self.dim, emb_type="entity", name="e", norm_method="norm")
        self.create_embedding((self.dim, self.dim), emb_type="relation", name="rh")
        self.create_embedding((self.dim, self.dim), emb_type="relation", name="rt")

    def _calc(self, h, rh, rt, t):
        # Eq. (3).
        hrh = torch.bmm(rh, h.view(-1, self.dim, 1)).view(-1, self.dim)
        trt = torch.bmm(rt, t.view(-1, self.dim, 1)).view(-1, self.dim)
        return -torch.linalg.norm(hrh - trt, dim=-1, ord=self.pnorm)

    def return_score(self, is_predict=False):
        (head_emb, rel_emb, tail_emb) = self.current_batch

        h = head_emb["e"]
        t = tail_emb["e"]
        rh, rt = rel_emb["rh"], rel_emb["rt"]

        return self._calc(h, rh, rt, t)
