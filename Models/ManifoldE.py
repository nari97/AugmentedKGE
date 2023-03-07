import torch
from Models.Model import Model


class ManifoldE(Model):
    """
    Han Xiao, Minlie Huang, Xiaoyan Zhu: From One Point to a Manifold: Knowledge Graph Embedding for Precise Link
        Prediction. IJCAI 2016: 1315-1321.
    """
    def __init__(self, ent_total, rel_total, dim, norm=1, variant='hyperplane'):
        """
            dim (int): Number of dimensions for embeddings
            norm (int): L1 or L2 norm. Default: 2
            variant can be either hyperplane or sphere. It seems hyperplane achieves better results.
        """
        super(ManifoldE, self).__init__(ent_total, rel_total)
        self.dim = dim
        self.pnorm = norm
        self.variant = variant

    def get_default_loss(self):
        # Eq. (3).
        return 'margin'

    def get_score_sign(self):
        # It is a distance (norm).
        return -1

    def initialize_model(self):
        self.create_embedding(self.dim, emb_type="entity", name="e")
        self.create_embedding(self.dim, emb_type="relation", name="rh")
        # Only for this variant.
        if self.variant == 'hyperplane':
            self.create_embedding(self.dim, emb_type="relation", name="rt")
        self.create_embedding(1, emb_type="relation", name="dr")

    def _calc(self, h, rh, t, rt, dr):
        # In the experiments, linear kernel achieves best results. We use it then.

        if self.variant == 'hyperplane':
            # See Hyperplane paragraph.
            m_hrt = torch.linalg.norm((h + rh) * (t + rt), dim=-1, ord=self.pnorm)

        if self.variant == 'sphere':
            # Eq. (2).
            m_hrt = torch.sum(h*h, dim=-1) + torch.sum(t*t, dim=-1) + torch.sum(rh*rh, dim=-1) -\
                2*torch.sum(h*t, dim=-1) - 2*torch.sum(rh*t, dim=-1) - 2*torch.sum(rh*h, dim=-1)

        # Eq. (1).
        return torch.pow(m_hrt - torch.pow(dr, 2).flatten(), 2)

    def return_score(self, is_predict=False):
        (head_emb, rel_emb, tail_emb) = self.current_batch

        h = head_emb["e"]
        t = tail_emb["e"]
        rh, rt, dr = rel_emb["rh"], None, rel_emb["dr"]
        if self.variant == 'hyperplane':
            rt = rel_emb["rt"]

        return self._calc(h, rh, t, rt, dr)
