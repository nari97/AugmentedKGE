import torch
from Models.Model import Model


class DistMult(Model):
    """
    Bishan Yang, Wen-tau Yih, Xiaodong He, Jianfeng Gao, Li Deng: Embedding Entities and Relations for Learning and
        Inference in Knowledge Bases. ICLR (Poster) 2015
    """
    def __init__(self, ent_total, rel_total, dim, variant='notanh'):
        """
            dim (int): Number of dimensions for embeddings
            variant can be either notanh or tanh. Indicates whether tanh should be used to project entity embeddings
                (see Table 4 in the paper, DistMult-tanh).
        """
        super(DistMult, self).__init__(ent_total, rel_total)
        self.dim = dim
        self.variant = variant

    def get_default_loss(self):
        # Eq. (3).
        return 'margin'

    def get_score_sign(self):
        # "...encourages the scores of positive relationships (triplets) to be higher than the scores of any negative
        #   relationships (triplets)."
        return 1

    def initialize_model(self):
        # From the paper: "The entity vectors are renormalized to have unit length after each gradient step (it is an
        #   effective technique that empirically improved all the models)."
        self.create_embedding(self.dim, emb_type="entity", name="e", norm_method="norm")
        # From the paper: "For the relation parameters, we used standard L2 regularization."
        self.create_embedding(self.dim, emb_type="relation", name="r", reg=True)
        
    def _calc(self, h, r, t):
        if self.variant == 'tanh':
            h, t = torch.tanh(h), torch.tanh(t)
        # Eq. (2).
        return torch.sum(h * r * t, -1)

    def return_score(self, is_predict=False):
        (head_emb, rel_emb, tail_emb) = self.current_batch

        h = head_emb["e"]
        t = tail_emb["e"]
        r = rel_emb["r"]

        return self._calc(h, r, t)
