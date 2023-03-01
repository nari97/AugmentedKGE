import torch
from Models.Model import Model


class CrossE(Model):
    """
    Wen Zhang, Bibek Paudel, Wei Zhang, Abraham Bernstein, Huajun Chen: Interaction Embeddings for Prediction and
        Explanation in Knowledge Graphs. WSDM 2019: 96-104.
    """
    def __init__(self, ent_total, rel_total, dim, apply_sigmoid=False, variant='interactions'):
        """
            dim (int): Number of dimensions for embeddings
            apply_sigmoid (Bool): Whether sigmoid must be applied to scores during training. Note that BCEWithLogitsLoss
                already applies sigmoid, so, if this is the loss function used, apply_sigmoid must be set to False. If
                a different loss function is applied, then apply_sigmoid should be set to True.
            variant can be either interactions or nointeractions (CrossES).
        """
        super(CrossE, self).__init__(ent_total, rel_total)
        self.dim = dim
        self.apply_sigmoid = apply_sigmoid
        self.variant = variant

    def get_default_loss(self):
        # Loss function after Eq. (8).
        return 'bce'

    def initialize_model(self):
        # Section 3 and loss function after Eq. (8).
        self.create_embedding(self.dim, emb_type="entity", name="e", reg=True)
        self.create_embedding(self.dim, emb_type="relation", name="r", reg=True)
        if self.variant == 'interactions':
            self.create_embedding(self.dim, emb_type="relation", name="c", reg=True)
        # After Eq. (5). Section 5.1.2: "b is initialized to zero."
        self.create_embedding(self.dim, emb_type="global", name="b",
                              init_method="uniform", init_params=[0, 0], reg=True)

    def _calc(self, h, r, c, t, b, is_predict=False):
        # Eq. (7) except sigmoid.
        scores = torch.sum(torch.tanh((c * h) + (c * h * r) + b) * t, -1)

        # Apply sigmoid when predicting or when indicated by apply_sigmoid.
        if is_predict or self.apply_sigmoid:
            scores = torch.sigmoid(scores)

        return scores

    def return_score(self, is_predict=False):
        (head_emb, rel_emb, tail_emb) = self.current_batch

        h = head_emb["e"]
        t = tail_emb["e"]
        r = rel_emb["r"]
        if self.variant == 'interactions':
            c = rel_emb["c"]
        elif self.variant == 'nointeractions':
            c = torch.ones_like(r)
        b = self.current_global_embeddings["b"]

        return self._calc(h, r, c, t, b, is_predict)
