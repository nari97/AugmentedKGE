import math
import torch
from Models.Model import Model


class AprilE(Model):
    """
    Yuzhang Liu, Peng Wang, Yingtai Li, Yizhan Shao, Zhongkai Xu: AprilE: Attention with Pseudo Residual Connection
        for Knowledge Graph Embedding. COLING 2020: 508-518.
    """
    def __init__(self, ent_total, rel_total, dim, norm=2):
        """
            dim (int): Number of dimensions for embeddings
            norm (int): L1 or L2 norm. Default: 2
        """
        super(AprilE, self).__init__(ent_total, rel_total)
        self.dim = dim
        self.pnorm = norm

    def get_default_loss(self):
        # Eq. (9).
        return 'margin'

    def get_score_sign(self):
        # It is a norm.
        return -1

    def initialize_model(self):
        # From the paper: "AprilE divides embeddings into two equal-size partitions, first and second."
        self.create_embedding(self.dim, emb_type="entity", name="e_first")
        self.create_embedding(self.dim, emb_type="entity", name="e_second")
        self.create_embedding(self.dim, emb_type="relation", name="r_first")
        self.create_embedding(self.dim, emb_type="relation", name="r_second")

        # From the paper: "AprilE applies the self-attention mechanism to capture the dependency of a triple."
        # w_k and w_q are the weights, b_k and b_q are the bias terms.
        self.create_embedding((self.dim, self.dim), emb_type="global", name="w_k")
        self.create_embedding(self.dim, emb_type="global", name="b_k")
        self.create_embedding((self.dim, self.dim), emb_type="global", name="w_q")
        self.create_embedding(self.dim, emb_type="global", name="b_q")

    def _calc(self, h_f, h_s, r_f, r_s, t_f, t_s, w_q, b_q, w_k, b_k):
        batch_size = h_f.shape[0]

        # Attention mechanism:

        # Stack h_s, r_s, t_f (Eq. (1)).
        c = torch.stack((h_s, r_s, t_f), dim=1)

        # Queries and keys (Eq. (2)).
        q = torch.relu(torch.bmm(c, w_q.expand(batch_size, self.dim, self.dim)) + b_q)
        k = torch.relu(torch.bmm(c, w_k.expand(batch_size, self.dim, self.dim)) + b_k)

        # Get attention using softmax. (Eqs. (3), (4) and (5))
        # When dim=2, the summation of each row is one.
        a = torch.nn.functional.softmax(q * k, dim=2) * c

        # Un-stack a to get h_s_dot, r_s_dot and t_f_dot.
        h_s_dot, r_s_dot, t_f_dot = torch.unbind(a, dim=1)

        # Eqs. (6) and (7)
        return torch.linalg.norm(h_f + h_s_dot + r_f + r_s_dot - (t_s + t_f_dot), dim=-1, ord=self.pnorm)

    def return_score(self, is_predict=False):
        (head_emb, rel_emb, tail_emb) = self.current_batch

        h_f, h_s = head_emb["e_first"], head_emb["e_second"]
        t_f, t_s = tail_emb["e_first"], tail_emb["e_second"]
        r_f, r_s = rel_emb["r_first"], rel_emb["r_second"]

        w_k, b_k, w_q, b_q = self.current_global_embeddings["w_k"], self.current_global_embeddings["b_k"], \
            self.current_global_embeddings["w_q"], self.current_global_embeddings["b_q"]

        return self._calc(h_f, h_s, r_f, r_s, t_f, t_s, w_q, b_q, w_k, b_k)
