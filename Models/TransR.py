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
        super(TransR, self).__init__(ent_total, rel_total)
        self.dim_e = dim_e
        self.dim_r = dim_r
        self.pnorm = norm

    def get_default_loss(self):
        return 'margin'

    def initialize_model(self):
        self.create_embedding(self.dim_e, emb_type="entity", name="e")
        self.create_embedding(self.dim_r, emb_type="relation", name="r")
        self.create_embedding((self.dim_e, self.dim_r), emb_type="relation", name="mr")

        self.register_scale_constraint(emb_type="entity", name="e")
        self.register_scale_constraint(emb_type="relation", name="r")

    def get_er(self, e, mr):
        batch_size = e.shape[0]
        # Change e into a row matrix, multiply and get final result as dim_r
        return torch.matmul(e.view(batch_size, 1, -1), mr).view(batch_size, self.dim_r)

    def _calc(self, h, r, mr, t, is_predict):
        hr = self.get_er(h, mr)
        tr = self.get_er(t, mr)

        if not is_predict:
            self.onthefly_constraints.append(self.scale_constraint(hr))
            self.onthefly_constraints.append(self.scale_constraint(tr))

        return -torch.linalg.norm(hr + r - tr, ord=self.pnorm, dim=-1)

    def return_score(self, is_predict=False):
        (head_emb, rel_emb, tail_emb) = self.current_batch

        h = head_emb["e"]
        t = tail_emb["e"]
        r, mr = rel_emb["r"], rel_emb["mr"]

        return self._calc(h, r, mr, t, is_predict)
