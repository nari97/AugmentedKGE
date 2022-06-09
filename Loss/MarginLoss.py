import torch
import torch.nn as nn
from .Loss import Loss


class MarginLoss(Loss):

    def __init__(self, model, margin=1e-1, criterion=None, reg_type='L2'):
        super(MarginLoss, self).__init__(model, is_pairwise=True, reg_type=reg_type)
        self.loss = nn.MarginRankingLoss(margin)
        self.criterion = criterion

    def lossFn(self, p_score, n_score):
        # We wish the positives to have a larger value than negatives, this is because our predict
        #   function changes the sign of the score!
        targets = torch.ones_like(p_score)
        if self.criterion is not None:
            p_score, n_score = self.criterion(p_score), self.criterion(n_score)
        return self.loss(p_score, n_score, targets)
