import torch
from Models.Model import Model


class TransR(Model):
    def __init__(self, ent_total, rel_total, dim_e, dim_r, norm=2):
        """
        Args:
            ent_total (int): Total number of entities
            rel_total (int): Total number of relations
            dim_e (int): Number of dimensions for entity embeddings
            dim_r (int): Number of dimensions for relation embeddings
        """
        super(TransR, self).__init__(ent_total, rel_total, 0, "transr")

        self.dim_e = dim_e
        self.dim_r = dim_r

        self.pnorm = norm

        self.create_embedding(self.dim_e, emb_type="entity", name="e")
        self.create_embedding(self.dim_r, emb_type="relation", name="r")
        self.create_embedding((self.dim_e, self.dim_r), emb_type="relation", name="mr")

        self.register_scale_constraint(emb_type="entity", name="e", p=2)
        self.register_scale_constraint(emb_type="relation", name="r", p=2)
        self.register_custom_constraint(self.h_constraint)
        self.register_custom_constraint(self.t_constraint)

    def h_constraint(self, head_emb, rel_emb, tail_emb, epsilon=1e-5):
        h = head_emb["e"]
        mr = rel_emb["mr"]
        return torch.linalg.norm(self.get_er(h, mr), ord=2) - epsilon

    def t_constraint(self, head_emb, rel_emb, tail_emb, epsilon=1e-5):
        t = tail_emb["e"]
        mr = rel_emb["mr"]
        return torch.linalg.norm(self.get_er(t, mr), ord=2) - epsilon

    def get_er(self, e, mr):
        # Change e into a row matrix, multiply and get final result as dim_r
        return torch.matmul(e.view(e.shape[0], 1, -1), mr).view(mr.shape[0], self.dim_r)

    def _calc(self, h, r, mr, t):
        return -torch.linalg.norm(self.get_er(h, mr) + r - self.get_er(t, mr), ord=self.pnorm, dim=-1)

    def return_score(self, head_emb, rel_emb, tail_emb, is_predict=False):
        h = head_emb["e"]
        t = tail_emb["e"]
        r = rel_emb["r"]
        mr = rel_emb["mr"]

        return self._calc(h, r, mr, t)
