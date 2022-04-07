import torch.nn as nn
from .Loss import Loss


class SoftMarginLoss(Loss):

    def __init__(self, model, margin=None):
        super(SoftMarginLoss, self).__init__(model, is_pairwise=False)
        self.loss = nn.SoftMarginLoss()
        self.margin = margin

    def lossFn(self, scores, targets):
        if self.margin is not None:
            scores = self.margin * targets + scores
        return self.loss(scores, targets)
