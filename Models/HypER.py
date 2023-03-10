import torch
from Models.Model import Model


class HypER(Model):
    """
    Ivana Balazevic, Carl Allen, Timothy M. Hospedales: Hypernetwork Knowledge Graph Embeddings. ICANN (Workshop) 2019:
        553-565.
    It is very similar to ConvE.
    """
    def __init__(self, ent_total, rel_total, dim_e, dim_r, input_dropout=0.2, hidden_dropout=0.3,
                 feature_dropout=0.2, apply_sigmoid=False):
        """
            dim_e (int): Number of dimensions for entity embeddings
            dim_r (int): Number of dimensions for relation embeddings
            input_dropout (float): dropout rate of the convolutional network's input.
            hidden_dropout (float): dropout rate of the convolutional network's hidden layer.
            feature_dropout (float): dropout rate of the convolutional network's features.
            apply_sigmoid (Bool): Whether sigmoid must be applied to scores during training. Note that BCEWithLogitsLoss
                already applies sigmoid, so, if this is the loss function used, apply_sigmoid must be set to False. If
                a different loss function is applied, then apply_sigmoid should be set to True.
        """
        super(HypER, self).__init__(ent_total, rel_total)
        self.dim_e = dim_e
        self.dim_r = dim_r
        self.apply_sigmoid = apply_sigmoid

        # Check Section 4.1. It is confusing what l_f and n_f are; by default in the code, they are both equal to three.
        #   It seems that n_f is the hidden dimension and l_f = 1 x Number.
        self.filter_length, self.number_filters = 3, 3

        # This is the number of out channels of the convolution.
        self.hidden_dim = 32

        # Dropouts.
        self.input_drop = torch.nn.Dropout(input_dropout)
        self.hidden_drop = torch.nn.Dropout(hidden_dropout)
        self.conv_feature_drop = torch.nn.Dropout2d(feature_dropout)
        # Normalization of the convolutional network.
        self.norm_input = torch.nn.BatchNorm2d(1, dtype=torch.float64)
        self.norm_conv = torch.nn.BatchNorm2d(self.hidden_dim, dtype=torch.float64)
        self.norm_hidden = torch.nn.BatchNorm1d(self.dim_e, dtype=torch.float64)
        # Linear transformation: from hidden to dim_e. The original code computes (1-self.filter_length+1), but,
        #   assuming that self.filter_length=3, that is a negative number! Right?
        self.fc = torch.nn.Linear(self.dim_e*self.hidden_dim, self.dim_e, dtype=torch.float64)
        # Linear transformation: from dim_r to hidden.
        self.fcr = torch.nn.Linear(self.dim_r, self.hidden_dim*self.filter_length*self.number_filters,
                                   dtype=torch.float64)

    def get_default_loss(self):
        # Eq. (2).
        return 'bce'

    def get_score_sign(self):
        # It is a similarity
        return 1

    def initialize_model(self):
        # Entities and relations. See Eq. (1).
        self.create_embedding(self.dim_e, emb_type="entity", name="e")
        # This is the bias.
        # Initialize b as zero: https://github.com/ibalazevic/HypER/blob/master/HypER/models.py#L75
        self.create_embedding(1, emb_type="entity", name="b", init_method="uniform", init_params=[0, 0])
        self.create_embedding(self.dim_r, emb_type="relation", name="r")
        # Everything related to the convolutional network is in __init__.

    def _calc(self, h, r, t, b, is_predict):
        # All this is Eq. (1), but there are missing details like dropouts, so we are following the original code.
        # (https://github.com/ibalazevic/HypER/blob/master/HypER/models.py#L57)
        # The original code has many transformations back and forth. We have simplified them.
        batch_size = h.shape[0]
        # Reshape h.
        h_reshaped = h.view(-1, 1, 1, self.dim_e)
        # Apply normalization and dropout.
        h_reshaped = self.input_drop(self.norm_input(h_reshaped))
        # Apply linear transformation to r and convolution to h.
        r_transformed = self.fcr(r)
        r_transformed = r_transformed.view(-1, 1, self.filter_length, self.number_filters)
        # We permute first and second because conv2d is expecting batch_size channels in the second dimension.
        h_reshaped = h_reshaped.permute(1, 0, 2, 3)
        # We add padding so it does not fail.
        x = torch.nn.functional.conv2d(h_reshaped, r_transformed, groups=batch_size, padding='same')
        # Normalize and dropout.
        x = self.conv_feature_drop(self.norm_conv(x.view(batch_size, self.hidden_dim, 1, self.dim_e)))
        # Apply linear transformation with dropout and normalization.
        x = self.norm_hidden(self.hidden_drop(self.fc(x.view(batch_size, -1))))
        # Apply ReLu.
        x = torch.nn.functional.relu(x)

        # Multiply by tails and apply bias.
        scores = torch.sum(x * t, -1).view(batch_size, -1) + b

        if is_predict or self.apply_sigmoid:
            scores = torch.sigmoid(scores)

        return scores

    def return_score(self, is_predict=False):
        (head_emb, rel_emb, tail_emb) = self.current_batch

        h = head_emb["e"]
        # The bias is for all the entities (https://github.com/TimDettmers/ConvE/blob/master/model.py#L121). The
        #   original code does not use the tails coming in the batch. We are going to assume that the bias only applies
        #   to the tails in the batch.
        t, b = tail_emb["e"], tail_emb["b"]
        r = rel_emb["r"]

        return self._calc(h, r, t, b, is_predict)
