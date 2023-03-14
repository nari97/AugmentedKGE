import torch
from Models.Model import Model


class TransRDT(Model):
    """
    Liang Chang, Manli Zhu, Tianlong Gu, Chenzhong Bin, Junyan Qian, Ji Zhang: Knowledge Graph Embedding by Dynamic
        Translation. IEEE Access 5: 20898-20907 (2017).
    """
    def __init__(self, ent_total, rel_total, dim_e, dim_r, norm=2, z=.3):
        """
            ent_total (int): Total number of entities
            rel_total (int): Total number of relations
            dim_e (int): Number of dimensions for entity embeddings
            dim_r (int): Number of dimensions for relation embeddings
            norm (int): L1 or L2 norm. Default: 2 (Eq. (8).).
            z (float): The L2-norm of the alpha parameters. From the paper: "...the L2-norm of them are values from the
                set {.1, .2, .3}."
        """
        super(TransRDT, self).__init__(ent_total, rel_total)
        self.dim_e = dim_e
        self.dim_r = dim_r
        self.pnorm = norm
        self.z = z

    def get_default_loss(self):
        return 'margin'

    def get_score_sign(self):
        # It is a distance (norm).
        return -1

    def initialize_model(self):
        self.create_embedding(self.dim_e, emb_type="entity", name="e")
        self.create_embedding(self.dim_r, emb_type="relation", name="r")
        self.create_embedding((self.dim_e, self.dim_r), emb_type="relation", name="mr")

        self.create_embedding(self.dim_r, emb_type="entity", name="ehalpha")
        self.create_embedding(self.dim_r, emb_type="entity", name="etalpha")
        self.create_embedding(self.dim_r, emb_type="relation", name="ralpha")

        self.register_scale_constraint(emb_type="entity", name="e")
        self.register_scale_constraint(emb_type="relation", name="r")

        # Scale constraints of the alpha parameters.
        self.register_scale_constraint(emb_type="entity", name="ehalpha", z=self.z)
        self.register_scale_constraint(emb_type="entity", name="etalpha", z=self.z)
        self.register_scale_constraint(emb_type="relation", name="ralpha", z=self.z)

    def _calc(self, h, halpha, r, mr, ralpha, t, talpha, is_predict):
        def transfer(e):
            batch_size = e.shape[0]
            return torch.matmul(e.view(batch_size, 1, -1), mr).view(batch_size, self.dim_r)
        hr, tr = transfer(h), transfer(t)
        if not is_predict:
            self.onthefly_constraints.append(self.scale_constraint(hr))
            self.onthefly_constraints.append(self.scale_constraint(tr))
        return torch.linalg.norm((hr + halpha) + (r + ralpha) - (tr + talpha), dim=-1, ord=self.pnorm)

    def return_score(self, is_predict=False):
        (head_emb, rel_emb, tail_emb) = self.current_batch

        h, halpha = head_emb["e"], head_emb["ehalpha"]
        t, talpha = tail_emb["e"], tail_emb["etalpha"]
        r, mr, ralpha = rel_emb["r"], rel_emb["mr"], rel_emb["ralpha"]

        return self._calc(h, halpha, r, mr, ralpha, t, talpha, is_predict)
