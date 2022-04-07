import torch
import torch.nn as nn
from .Loss import Loss


class MarginLoss(Loss):

    def __init__(self, model, margin=1e-1, criterion=None):
        super(MarginLoss, self).__init__(model, is_pairwise=True)
        self.loss = nn.MarginRankingLoss(margin)
        self.criterion = criterion

    def lossFn(self, p_score, n_score):
        # We wish the positives to have a larger value than negatives, this is because our predict
        #   function changes the sign of the score!
        targets = torch.empty((len(p_score), 1))
        targets = nn.init.constant_(targets, 1)
        if p_score.is_cuda:
            targets = targets.cuda()
        if self.criterion is not None:
            p_score, n_score = self.criterion(p_score), self.criterion(n_score)
        return self.loss(p_score, n_score, targets)
