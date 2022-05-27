import torch
from Models.Model import Model


class TransA(Model):

    def __init__(self, ent_total, rel_total, dim):
        super(TransA, self).__init__(ent_total, rel_total)
        self.dim = dim

    def get_default_loss(self):
        return 'margin'

    def initialize_model(self):
        self.create_embedding(self.dim, emb_type="entity", name="e")
        self.create_embedding(self.dim, emb_type="relation", name="r")
        self.create_embedding((self.dim, self.dim), emb_type="relation", name="w")

        self.register_scale_constraint(emb_type="entity", name="e", p=2)
        self.register_scale_constraint(emb_type="relation", name="r", p=2)
        self.register_custom_constraint(self.frob_constraint)

    def get_matrix(self, w):
        # Make sure it is symmetric and positive.
        return torch.abs(torch.bmm(w, torch.transpose(w, 1, 2)))

    def frob_constraint(self, head_emb, rel_emb, tail_emb, epsilon=1e-5):
        return torch.linalg.matrix_norm(self.get_matrix(rel_emb["w"]), ord='fro') - epsilon

    def _calc(self, h, r, t, w):
        batch_size = h.shape[0]
        a = torch.abs(h+r-t)
        return torch.matmul(torch.matmul(a.view(batch_size, 1, -1), self.get_matrix(w)), a.view(batch_size, -1, 1))

    def return_score(self, is_predict=False):
        (head_emb, rel_emb, tail_emb) = self.current_batch

        h = head_emb["e"]
        t = tail_emb["e"]
        r = rel_emb["r"]
        w = rel_emb["w"]

        return self._calc(h, r, t, w)
