import torch
from Models.Model import Model


class TransA(Model):
    """
    Han Xiao, Minlie Huang, Xiaoyan Zhu: TransG : A Generative Model for Knowledge Graph Embedding. ACL (1) 2016.
    """
    def __init__(self, ent_total, rel_total, dim):
        """
            dim (int): Number of dimensions for embeddings
        """
        super(TransA, self).__init__(ent_total, rel_total)
        self.dim = dim

    def get_default_loss(self):
        # Eq. (7).
        return 'margin'

    def initialize_model(self):
        # See Eq. (7) for regularization.
        self.create_embedding(self.dim, emb_type="entity", name="e", reg=True)
        self.create_embedding(self.dim, emb_type="relation", name="r", reg=True)
        # The paper seems to imply that the matrix can be calculated from the embeddings (see Eq. 8). That requires
        #   computations at the end of the loss function. We use a matrix parameter.
        # See Eq. (7) for regularization.
        self.create_embedding((self.dim, self.dim), emb_type="relation", name="w", reg=True,
                              reg_params={"norm": torch.linalg.matrix_norm, "p": 'fro', "dim": (-2, -1)})

    def _calc(self, h, r, t, w):
        batch_size = h.shape[0]
        # Eq. (2).
        a = torch.abs(h + r - t)
        # Make sure it is symmetric and positive. From the paper: "...we set all the negative entries of Wr to zero."
        w = torch.clamp(torch.bmm(w, torch.transpose(w, 1, 2)), min=0)
        # Eq. (2).
        return torch.matmul(torch.matmul(a.view(batch_size, 1, -1), w), a.view(batch_size, -1, 1))

    def return_score(self, is_predict=False):
        (head_emb, rel_emb, tail_emb) = self.current_batch

        h = head_emb["e"]
        t = tail_emb["e"]
        r, w = rel_emb["r"], rel_emb["w"]

        return self._calc(h, r, t, w)
