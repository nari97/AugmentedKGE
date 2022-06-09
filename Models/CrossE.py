import torch
from Models.Model import Model


class CrossE(Model):

    def __init__(self, ent_total, rel_total, dim):
        """
        Args:
            ent_total (int): Total number of entities
            rel_total (int): Total number of relations
            dim (int): Number of dimensions for embeddings
        """
        super(CrossE, self).__init__(ent_total, rel_total)
        self.dim = dim

    def get_default_loss(self):
        return 'bce'

    def initialize_model(self):
        self.create_embedding(self.dim, emb_type="entity", name="e")
        self.create_embedding(self.dim, emb_type="relation", name="r")
        self.create_embedding(self.dim, emb_type="relation", name="c")
        self.create_embedding(self.dim, emb_type="global", name="b")

        self.register_scale_constraint(emb_type="entity", name="e")
        self.register_scale_constraint(emb_type="relation", name="r")
        self.register_scale_constraint(emb_type="relation", name="c")
        self.register_scale_constraint(emb_type="global", name="b")

    def _calc(self, h, r, t, c, b, is_predict=False):
        scores = torch.sum(torch.tanh((c * h) + (c * h * r) + b) * t, -1)

        # Apply only sigmoid when predicting.
        if is_predict:
            scores = torch.sigmoid(scores)

        return scores

    def return_score(self, is_predict=False):
        (head_emb, rel_emb, tail_emb) = self.current_batch

        h = head_emb["e"]
        t = tail_emb["e"]
        r = rel_emb["r"]
        c = rel_emb["c"]
        b = self.current_global_embeddings["b"]

        return self._calc(h, r, t, c, b, is_predict)
