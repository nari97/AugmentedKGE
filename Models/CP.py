import torch
from Models.Model import Model


class CP(Model):
    """
    Timothée Lacroix, Nicolas Usunier, Guillaume Obozinski: Canonical Tensor Decomposition for Knowledge Base
        Completion. ICML 2018: 2869-2878.
    """
    def __init__(self, ent_total, rel_total, dim):
        """
            dim (int): Number of dimensions for embeddings
        """
        super(CP, self).__init__(ent_total, rel_total)
        self.dim = dim

    def get_default_loss(self):
        # Eq. (1). They differentiate between head and tail corruptions. We do not do that.
        return 'logsoftmax'

    def get_score_sign(self):
        # It is a similarity
        return 1

    def initialize_model(self):
        # It is the same as DistMult but uses two embeddings depending on head/tail.
        # All terms are regularized (see Eq. (2)). The paper mentions a regularization weight that we do not implement.
        #   It also suggests L3 regularization.
        self.create_embedding(self.dim, emb_type="entity", name="eh", reg=True)
        self.create_embedding(self.dim, emb_type="entity", name="et", reg=True)
        self.create_embedding(self.dim, emb_type="relation", name="r", reg=True)
        
    def _calc(self, h, r, t):
        # See CP in Section 2.
        return torch.sum(h * r * t, -1)

    def return_score(self, is_predict=False):
        (head_emb, rel_emb, tail_emb) = self.current_batch

        h = head_emb["eh"]
        t = tail_emb["et"]
        r = rel_emb["r"]

        return self._calc(h, r, t)
