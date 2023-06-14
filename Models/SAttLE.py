import torch
from Models.Model import Model


class SAttLE(Model):
    """
    Peyman Baghershahi, Reshad Hosseini, Hadi Moradi: Self-attention presents low-dimensional knowledge graph embeddings
        for link prediction. Knowl. Based Syst. 260: 110124 (2023).
    """
    def __init__(self, ent_total, rel_total, dim, apply_sigmoid=False, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048):
        """
            dim (int): Number of dimensions for embeddings
            apply_sigmoid (Bool): Whether sigmoid must be applied to scores during training. Note that BCEWithLogitsLoss
                already applies sigmoid, so, if this is the loss function used, apply_sigmoid must be set to False. If
                a different loss function is applied, then apply_sigmoid should be set to True.
            nhead, num_encoder_layers, num_decoder_layers, dim_feedforward: These are for the transformer.
        """
        super(SAttLE, self).__init__(ent_total, rel_total)
        self.dim = dim
        self.apply_sigmoid = apply_sigmoid
        # dim*2 because we stack h and r.
        self.transformer = torch.nn.Transformer(d_model=self.dim*2, nhead=nhead, num_encoder_layers=num_encoder_layers,
                                                num_decoder_layers=num_decoder_layers, dim_feedforward=dim_feedforward,
                                                dtype=torch.float64)
        # This will give us the scores.
        self.fc = torch.nn.Linear(self.dim*2, 1, dtype=torch.float64)

    def get_default_loss(self):
        # See Eq. (8).
        return 'bce'

    def get_score_sign(self):
        # It uses log-likelihood to learn the parameters.
        return 1

    def initialize_model(self):
        # Section 3.2.
        self.create_embedding(self.dim, emb_type="entity", name="e")
        self.create_embedding(self.dim, emb_type="relation", name="r")

    # Check: https://github.com/pbaghershahi/SAttLE/blob/main/model.py#L214
    def _calc(self, h, r, t, is_predict=False):
        # Transform: encoding stack of h and r, and decoding t duplicated.
        transformation = self.transformer(torch.hstack((h, r)), torch.hstack((t, t)))
        # In the paper, only the transformed r is used (Eq. (5)), which is the second part of the transformation. We use
        #   the whole vector here.
        scores = self.fc(transformation)

        # Apply sigmoid when predicting or when indicated by apply_sigmoid.
        if is_predict or self.apply_sigmoid:
            scores = torch.sigmoid(scores)

        return scores

    def return_score(self, is_predict=False):
        (head_emb, rel_emb, tail_emb) = self.current_batch

        h = head_emb["e"]
        t = tail_emb["e"]
        r = rel_emb["r"]

        return self._calc(h, r, t, is_predict)
