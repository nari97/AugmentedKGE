import math
import torch
from Models.Model import Model


class SpacE(Model):
    """
    Mojtaba Nayyeri, Chengjin Xu, Sahar Vahdati, Nadezhda Vassilyeva, Emanuel Sallinger, Hamed Shariat Yazdi, Jens
        Lehmann: Fantastic Knowledge Graph Embeddings and How to Find the Right Space for Them. ISWC (1) 2020: 438-455.
    """
    def __init__(self, ent_total, rel_total, dim, norm=1):
        """
            dim (int): Number of dimensions for embeddings
            norm (int): L1 or L2 norm. Default: 1, as in RotatE, but unclear from the paper.
        """
        super(SpacE, self).__init__(ent_total, rel_total)
        self.dim = dim
        self.pnorm = norm

    def get_default_loss(self):
        # From the paper: "We train our model by using the RotatE loss [19]."
        return 'soft_margin'

    def get_score_sign(self):
        # It is a distance (norm).
        return -1

    def initialize_model(self):
        # See Eqs. (3) and (4).
        self.create_embedding(self.dim, emb_type="entity", name="e")
        self.create_embedding(self.dim, emb_type="relation", name="r")
        # Initialization is not mentioned in the paper; we used the same as RotatE.
        self.create_embedding(self.dim, emb_type="entity", name="e_phase_head",
                              init_method="uniform", init_params=[0, 2 * math.pi])
        self.create_embedding(self.dim, emb_type="entity", name="e_phase_tail",
                              init_method="uniform", init_params=[0, 2 * math.pi])
        # No constraints are mentioned.

    def _calc(self, h, h_phase, r, t, t_phase):
        # Use polar form for both phases.
        h_rotation = torch.polar(torch.ones_like(h_phase), h_phase)
        t_rotation = torch.polar(torch.ones_like(t_phase), -t_phase)
        # These are translated into complex numbers.
        hc = torch.view_as_complex(torch.stack((h, torch.zeros_like(h)), dim=-1))
        tc = torch.view_as_complex(torch.stack((t, torch.zeros_like(t)), dim=-1))
        rc = torch.view_as_complex(torch.stack((r, torch.zeros_like(r)), dim=-1))
        # Eq. (4).
        return torch.linalg.norm((hc * h_rotation) + rc - (tc * t_rotation), dim=-1, ord=self.pnorm)

    def return_score(self, is_predict=False):
        (head_emb, rel_emb, tail_emb) = self.current_batch

        h, h_phase = head_emb["e"], head_emb["e_phase_head"]
        t, t_phase = tail_emb["e"], tail_emb["e_phase_tail"]
        r = rel_emb["r"]

        return self._calc(h, h_phase, r, t, t_phase)
