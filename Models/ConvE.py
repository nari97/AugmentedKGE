import math
import torch
from Models.Model import Model


class ConvE(Model):
    """
    Tim Dettmers, Pasquale Minervini, Pontus Stenetorp, Sebastian Riedel: Convolutional 2D Knowledge Graph Embeddings.
        AAAI 2018: 1811-1818.
    """
    def __init__(self, ent_total, rel_total, dim, dim_reshaping=None, input_dropout=0.2, hidden_dropout=0.3,
                 feature_dropout=0.2, apply_sigmoid=False):
        """
            dim (int): Dimensions for embeddings
            dim_reshaping (int): 2D-reshaping dimension.
            input_dropout (float): dropout rate of the convolutional network's input.
            hidden_dropout (float): dropout rate of the convolutional network's hidden layer.
            feature_dropout (float): dropout rate of the convolutional network's features.
            apply_sigmoid (Bool): Whether sigmoid must be applied to scores during training. Note that BCEWithLogitsLoss
                already applies sigmoid, so, if this is the loss function used, apply_sigmoid must be set to False. If
                a different loss function is applied, then apply_sigmoid should be set to True.
        """
        super(ConvE, self).__init__(ent_total, rel_total)
        self.dim = dim
        self.dim_w = dim_reshaping
        self.apply_sigmoid = apply_sigmoid

        if self.dim_w is None:
            # In their implementation, they use dim=200 and dim_reshaping=20 (See:
            #   https://github.com/TimDettmers/ConvE/blob/master/main.py#L175). We will use 20% of dim by default.
            self.dim_w = math.ceil(self.dim * .2)

        # According to the paper, vectors are 2D-reshaped into dim = dim_w times dim_h.
        self.dim_h = self.dim // self.dim_w
        # We need to adjust dim; there are issues when the indicated dim is odd.
        self.dim = self.dim_w * self.dim_h
        # When we stack embeddings, the resulting dimension is 2*dim_w. The kernel size in the convolution cannot be
        #   larger than 2*dim_w.
        kernel = (3, 3) # By default, 3x3
        if 2*self.dim_w < 3:
            kernel = (self.dim_w, self.dim_w)
        # This is the number of out channels of the convolution.
        hidden_dim = 32

        # Convolutional network; by default, we use bias.
        self.conv = torch.nn.Conv2d(1, hidden_dim, kernel, 1, 0, dtype=torch.float64)
        # Dropouts.
        self.input_drop = torch.nn.Dropout(input_dropout)
        self.hidden_drop = torch.nn.Dropout(hidden_dropout)
        self.conv_feature_drop = torch.nn.Dropout2d(feature_dropout)
        # Normalization of the convolutional network.
        self.norm_input = torch.nn.BatchNorm2d(1, dtype=torch.float64)
        self.norm_conv = torch.nn.BatchNorm2d(hidden_dim, dtype=torch.float64)
        self.norm_hidden = torch.nn.BatchNorm1d(self.dim, dtype=torch.float64)
        # Linear transformation: from unstacked to dim.
        self.fc = torch.nn.Linear(self.dim_w * 2 * self.dim_h * hidden_dim, self.dim, dtype=torch.float64)

    def get_default_loss(self):
        # Eq. (2).
        return 'bce'

    def get_score_sign(self):
        # It is a similarity
        return 1

    def initialize_model(self):
        # Entities and relations. See Eq. (1).
        self.create_embedding(self.dim, emb_type="entity", name="e")
        # This is the bias.
        # Initialize b as zero: https://github.com/TimDettmers/ConvE/blob/master/model.py#L95
        self.create_embedding(1, emb_type="entity", name="b", init_method="uniform", init_params=[0, 0])
        self.create_embedding(self.dim, emb_type="relation", name="r")
        # Everything related to the convolutional network is in __init__.

    def _calc(self, h, r, t, b, is_predict):
        # All this is Eq. (1), but there are missing details like dropouts, so we are following the original code.
        batch_size = h.shape[0]
        # Reshape h and r.
        h_reshaped = h.view(-1, 1, self.dim_w, self.dim_h)
        r_reshaped = r.view(-1, 1, self.dim_w, self.dim_h)
        # Stack reshaped inputs.
        stacked_inputs = torch.cat([h_reshaped, r_reshaped], 2)
        # Apply normalization and dropouts.
        stacked_inputs = self.input_drop(self.norm_input(stacked_inputs))
        # Apply convolutional network and normalization.
        x = self.norm_conv(self.conv(stacked_inputs))
        # Apply ReLu and dropout.
        x = self.conv_feature_drop(torch.nn.functional.relu(x))
        # Unstack and apply linear transformation with dropout and normalization.
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
