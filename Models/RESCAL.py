import torch
from Models.Model import Model


# TODO: Work on this one!
class RESCAL(Model):
    """
    Maximilian Nickel, Volker Tresp, Hans-Peter Kriegel: A Three-Way Model for Collective Learning on Multi-Relational
        Data. ICML 2011: 809-816.
    """
    def __init__(self, ent_total, rel_total, dim):
        super(RESCAL, self).__init__(ent_total, rel_total)
        self.dim = dim

    def get_default_loss(self):
        # The original paper did not use SGD; we assume margin.
        # TODO Eq. (3) uses a custom loss function and Eq. (4) proposes regularization.
        return 'margin'

    def initialize_model(self):
        self.create_embedding(self.dim, emb_type="entity", name="e")
        self.create_embedding((self.dim, self.dim), emb_type="relation", name="r")

    def _calc(self, h, r, t):
        batch_size = h.shape[0]
        return torch.matmul(torch.matmul(h.view(batch_size, 1, -1), r), t.view(batch_size, -1, 1))

    def return_score(self, is_predict=False):
        (head_emb, rel_emb, tail_emb) = self.current_batch

        h = head_emb["e"]
        t = tail_emb["e"]
        r = rel_emb["r"]

        return self._calc(h, r, t)
