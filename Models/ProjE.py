import torch
from Models.Model import Model


class ProjE(Model):
    """
    Baoxu Shi, Tim Weninger: ProjE: Embedding Projection for Knowledge Graph Completion. AAAI 2017: 1236-1242.
    """
    def __init__(self, ent_total, rel_total, dim, apply_sigmoid=False, hidden_dropout=0.3, variant='pointwise'):
        """
            dim (int): Number of dimensions for embeddings
            apply_sigmoid (Bool): Whether sigmoid must be applied to scores during training. Note that BCEWithLogitsLoss
                already applies sigmoid, so, if this is the loss function used, apply_sigmoid must be set to False. If
                a different loss function is applied, then apply_sigmoid should be set to True.
            variant can be pointwise or listwise.
        """
        super(ProjE, self).__init__(ent_total, rel_total)
        self.dim = dim
        self.apply_sigmoid = apply_sigmoid
        self.variant = variant
        self.dropout = torch.nn.Dropout(hidden_dropout)

    def get_default_loss(self):
        if self.variant == 'pointwise':
            # Eq. (6).
            return 'bce'
        if self.variant == 'listwise':
            # Eq. (8).
            return 'logsoftmax'

    def get_score_sign(self):
        # It is a similarity.
        return 1

    def initialize_model(self):
        # See Eqs. (4) and (5).
        # "we apply an L1 regularizer to all parameters in ProjE"
        # "ProjE only increases the number of parameters by 5k + 1, where 1, 4k, and k are introduced as the projection
        #   bias, combination weights, and combination bias respectively."
        # We have 1 (bp) + 2k (de and dr) + k (bc). Where are the other 2k?
        self.create_embedding(self.dim, emb_type="entity", name="e", reg=True)
        self.create_embedding(self.dim, emb_type="relation", name="r", reg=True)
        self.create_embedding(self.dim, emb_type="global", name="de", reg=True)
        self.create_embedding(self.dim, emb_type="global", name="dr", reg=True)
        self.create_embedding(self.dim, emb_type="global", name="bc", reg=True)
        self.create_embedding(1, emb_type="global", name="bp", reg=True)

    def _calc(self, h, r, t, de, dr, bc, bp, is_predict):
        if self.variant == 'pointwise':
            g = lambda x: x  # Do nothing! Sigmoid is applied at the end, if needed.
        if self.variant == 'listwise':
            g = lambda x: torch.nn.functional.softmax(x, dim=1)

        # Eq. (4).
        combination = de * h + dr * r + bc
        # Eq. (5) where Wc is substituted by t.
        # "[we apply] a dropout layer on top of the combination operator to prevent over-fitting."
        scores = g(torch.sum(t * torch.tanh(self.dropout(combination)), -1) + bp)

        # Apply only sigmoid when predicting.
        if self.variant == 'pointwise' and (is_predict or self.apply_sigmoid):
            scores = torch.sigmoid(scores)

        return scores

    def return_score(self, is_predict=False):
        (head_emb, rel_emb, tail_emb) = self.current_batch

        h = head_emb["e"]
        t = tail_emb["e"]
        r = rel_emb["r"]
        de, dr, bc, bp = self.current_global_embeddings["de"], self.current_global_embeddings["dr"], \
            self.current_global_embeddings["bc"], self.current_global_embeddings["bp"]

        return self._calc(h, r, t, de, dr, bc, bp, is_predict)
