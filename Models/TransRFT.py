import torch
from Models.Model import Model


class TransRFT(Model):
    """
    Jun Feng, Minlie Huang, Mingdong Wang, Mantong Zhou, Yu Hao, Xiaoyan Zhu: Knowledge Graph Embedding by Flexible
        Translation. KR 2016: 557-560.
    """
    def __init__(self, ent_total, rel_total, dim_e, dim_r):
        """
            ent_total (int): Total number of entities
            rel_total (int): Total number of relations
            dim_e (int): Number of dimensions for entity embeddings
            dim_r (int): Number of dimensions for relation embeddings
        """
        super(TransRFT, self).__init__(ent_total, rel_total)
        self.dim_e = dim_e
        self.dim_r = dim_r

    def get_default_loss(self):
        # Eq. (1).
        return 'margin'

    def get_score_sign(self):
        # It is a similarity. From the paper: "...we design a new function to score the compatibility of a triple by the
        #   inner product between the sum of head entity vector and relation vector h + r and tail vector t instead of
        #   using the Manhattan/Euclidean distance..."
        return 1

    def initialize_model(self):
        self.create_embedding(self.dim_e, emb_type="entity", name="e")
        self.create_embedding(self.dim_r, emb_type="relation", name="r")
        self.create_embedding((self.dim_e, self.dim_r), emb_type="relation", name="mr")
        self.register_scale_constraint(emb_type="entity", name="e")
        self.register_scale_constraint(emb_type="relation", name="r")

    def _calc(self, h, r, mr, t, is_predict):
        def transfer(e):
            batch_size = e.shape[0]
            return torch.matmul(e.view(batch_size, 1, -1), mr).view(batch_size, self.dim_r)
        ht, tt = transfer(h), transfer(t)
        if not is_predict:
            self.onthefly_constraints.append(self.scale_constraint(ht))
            self.onthefly_constraints.append(self.scale_constraint(tt))
        return torch.sum((ht + r) * tt + ht * (tt - r), -1)

    def return_score(self, is_predict=False):
        (head_emb, rel_emb, tail_emb) = self.current_batch

        h = head_emb["e"]
        t = tail_emb["e"]
        r, mr = rel_emb["r"], rel_emb["mr"]

        return self._calc(h, r, mr, t, is_predict)
