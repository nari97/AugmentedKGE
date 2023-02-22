import math
import torch
from Models.Model import Model


class pRotatE(Model):
    """
    Zhiqing Sun, Zhi-Hong Deng, Jian-Yun Nie, Jian Tang: RotatE: Knowledge Graph Embedding by Relational Rotation in
        Complex Space. ICLR (Poster) 2019.
    This is a variant of RotatE.
    """
    def __init__(self, ent_total, rel_total, dim, norm=1):
        """
            dim (int): Number of dimensions for embeddings
            norm (int): L1 or L2 norm. Default: 1. See footnote 2 in the paper.
        """
        super(pRotatE, self).__init__(ent_total, rel_total)
        self.dim = dim
        self.pnorm = norm

    def get_default_loss(self):
        # Eq. (4).
        return 'soft_margin'

    def initialize_model(self):
        # Only phases.
        self.create_embedding(self.dim, emb_type="entity", name="e_phase")
        self.create_embedding(self.dim, emb_type="relation", name="r_phase",
                              init_method="uniform", init_params=[0, 2 * math.pi])

        # https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding/blob/master/codes/model.py#L59
        # c is a modulus, so it must be positive.
        self.create_embedding(1, emb_type="global", name="c", norm_method="rescaling", norm_params={"a": 0})

    def _calc(self, h_phase, r_phase, t_phase, c):
        # See Section 4.1 and Eq. (17).
        return - 2 * c * torch.linalg.norm(torch.sin((h_phase+r_phase-t_phase)/2), dim=-1, ord=self.pnorm)

    def return_score(self, is_predict=False):
        (head_emb, rel_emb, tail_emb) = self.current_batch

        h_phase = head_emb["e_phase"]
        t_phase = tail_emb["e_phase"]
        r_phase = rel_emb["r_phase"]

        c = self.current_global_embeddings["c"].item()

        return self._calc(h_phase, r_phase, t_phase, c)
