import torch
from Models.Model import Model


class CyclE(Model):
    def __init__(self, ent_total, rel_total, dim, omega=10):
        super(CyclE, self).__init__(ent_total, rel_total)
        self.dim = dim
        # This is the hyperparameter that controls the space volume: how many entities can be related to other entities.
        # As a result, the value needs to be large.
        self.omega = omega

    def get_default_loss(self):
        return 'margin'

    def initialize_model(self):
        self.create_embedding(self.dim, emb_type="entity", name="e", norm_method="norm")
        self.create_embedding(self.dim, emb_type="relation", name="r")
        self.create_embedding(1, emb_type="global", name="a")
        self.create_embedding(1, emb_type="global", name="b")
        self.create_embedding(1, emb_type="global", name="g")

    def _calc(self, h, r, t, a, b, g):
        # The paper does not specify any function, but we need to get a single score.
        return torch.sum(a * torch.sin(self.omega * (h + r - t) + b) + g, dim=1)

    def return_score(self, is_predict=False):
        (head_emb, rel_emb, tail_emb) = self.current_batch

        h = head_emb["e"]
        t = tail_emb["e"]
        r = rel_emb["r"]

        a, b, g = self.current_global_embeddings["a"], \
            self.current_global_embeddings["b"], self.current_global_embeddings["g"]

        return self._calc(h, r, t, a, b, g)
