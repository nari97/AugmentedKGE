import torch
import torch.nn as nn
from .Loss import Loss


class BCELoss(Loss):

    def __init__(self, model, with_logits=False, margin=None):
        super(BCELoss, self).__init__(model, is_pairwise=False)

        self.margin = margin

        if not with_logits:
            self.loss = nn.BCELoss()
        else:
            self.loss = nn.BCEWithLogitsLoss()

    def lossFn(self, scores, targets):
        if self.margin is not None:
            # Apply margin to scores.
            scores = scores + targets * self.margin

        # Change from -1 to 0.
        targets[targets == -1] = 0
        # TODO CUDA for targets?
        return self.loss(scores, targets.to(torch.float64))
