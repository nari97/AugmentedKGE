import math
import torch
from Models.Model import Model


class AprilE(Model):
    def __init__(self, ent_total, rel_total, dim, norm=2):
        super(AprilE, self).__init__(ent_total, rel_total)
        self.dim = dim
        self.pnorm = norm

    def get_default_loss(self):
        return 'margin'

    def initialize_model(self):
        self.create_embedding(self.dim, emb_type="entity", name="e_first")
        self.create_embedding(self.dim, emb_type="entity", name="e_second")
        self.create_embedding(self.dim, emb_type="relation", name="r_first")
        self.create_embedding(self.dim, emb_type="relation", name="r_second")

        self.create_embedding((self.dim, self.dim), emb_type="global", name="w_k")
        self.create_embedding(self.dim, emb_type="global", name="b_k")
        self.create_embedding((self.dim, self.dim), emb_type="global", name="w_q")
        self.create_embedding(self.dim, emb_type="global", name="b_q")

    def _calc(self, h_f, h_s, r_f, r_s, t_f, t_s, w_q, b_q, w_k, b_k):
        batch_size = h_f.shape[0]

        # Attention mechanism:

        # Stack h_s, r_s, t_f.
        c = torch.stack((h_s, r_s, t_f), dim=1)

        # Queries and keys.
        q = torch.relu(torch.bmm(c, w_q.expand(batch_size, self.dim, self.dim)) + b_q)
        k = torch.relu(torch.bmm(c, w_k.expand(batch_size, self.dim, self.dim)) + b_k)

        # Get attention using softmax. I think we want to apply this to the last dimension.
        a = torch.nn.functional.softmax(q * k, dim=2) * c

        # Un-stack a to get h_s_dot, r_s_dot and t_f_dot.
        h_s_dot, r_s_dot, t_f_dot = torch.unbind(a, dim=1)

        return -torch.linalg.norm(h_f + h_s_dot + r_f + r_s_dot - (t_s + t_f_dot), dim=-1, ord=self.pnorm)

    def return_score(self, is_predict=False):
        (head_emb, rel_emb, tail_emb) = self.current_batch

        h_f, h_s = head_emb["e_first"], head_emb["e_second"]
        t_f, t_s = tail_emb["e_first"], tail_emb["e_second"]
        r_f, r_s = rel_emb["r_first"], rel_emb["r_second"]

        w_k, b_k, w_q, b_q = self.current_global_embeddings["w_k"], self.current_global_embeddings["b_k"], \
            self.current_global_embeddings["w_q"], self.current_global_embeddings["b_q"]

        return self._calc(h_f, h_s, r_f, r_s, t_f, t_s, w_q, b_q, w_k, b_k)
