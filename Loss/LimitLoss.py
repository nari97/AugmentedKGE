import torch
import torch.nn as nn
from .Loss import Loss


class LimitLoss(Loss):

    def __init__(self, model, margin_p=1e-1, margin_n=1e-1, alpha=.8, criterion=None, reg_type='L2'):
        super(LimitLoss, self).__init__(model, is_pairwise=True, reg_type=reg_type)
        self.loss_p = nn.MarginRankingLoss(margin_p)
        self.loss_n = nn.MarginRankingLoss(margin_n)
        self.alpha = alpha
        self.criterion = criterion

    def lossFn(self, p_score, n_score):
        # We wish the positives to have a larger value than negatives, this is because our predict
        #   function changes the sign of the score!
        targets_p, targets_n = torch.ones_like(p_score), torch.ones_like(n_score)
        zeros_p, zeros_n = torch.zeros_like(p_score), torch.zeros_like(n_score)
        if self.criterion is not None:
            p_score, n_score = self.criterion(p_score), self.criterion(n_score)
        return self.loss_p(p_score, zeros_p, targets_p) + self.alpha * self.loss_n(zeros_n, n_score, targets_n)
