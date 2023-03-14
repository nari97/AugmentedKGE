import torch
from Models.Model import Model


class TransEDT(Model):
    """
    Liang Chang, Manli Zhu, Tianlong Gu, Chenzhong Bin, Junyan Qian, Ji Zhang: Knowledge Graph Embedding by Dynamic
        Translation. IEEE Access 5: 20898-20907 (2017).
    """
    def __init__(self, ent_total, rel_total, dim, norm=2, z=.3):
        """
            dim (int): Number of dimensions for embeddings
            norm (int): L1 or L2 norm. Default: 2
            z (float): The L2-norm of the alpha parameters. From the paper: "...the L2-norm of them are values from the
                set {.1, .2, .3}."
        """
        super(TransEDT, self).__init__(ent_total, rel_total)
        self.dim = dim
        self.pnorm = norm
        self.z = z

    def get_default_loss(self):
        # Eq. (1).
        return 'margin'

    def get_score_sign(self):
        # It is a distance (norm).
        return -1

    def initialize_model(self):
        self.create_embedding(self.dim, emb_type="entity", name="e", norm_method="norm")
        self.create_embedding(self.dim, emb_type="relation", name="r")

        self.create_embedding(self.dim, emb_type="entity", name="ehalpha")
        self.create_embedding(self.dim, emb_type="entity", name="etalpha")
        self.create_embedding(self.dim, emb_type="relation", name="ralpha")

        # Scale constraints of the alpha parameters.
        self.register_scale_constraint(emb_type="entity", name="ehalpha", z=self.z)
        self.register_scale_constraint(emb_type="entity", name="etalpha", z=self.z)
        self.register_scale_constraint(emb_type="relation", name="ralpha", z=self.z)

    def _calc(self, h, halpha, r, ralpha, t, talpha):
        return torch.linalg.norm((h + halpha) + (r + ralpha) - (t + talpha), dim=-1, ord=self.pnorm)

    def return_score(self, is_predict=False):
        (head_emb, rel_emb, tail_emb) = self.current_batch

        h, halpha = head_emb["e"], head_emb["ehalpha"]
        t, talpha = tail_emb["e"], tail_emb["etalpha"]
        r, ralpha = rel_emb["r"], rel_emb["ralpha"]

        return self._calc(h, halpha, r, ralpha, t, talpha)
