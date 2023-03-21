import torch
import torch.nn as nn
from .Loss import Loss


class LogSoftmaxLoss(Loss):

    def __init__(self, model, reg_type='L2'):
        super(LogSoftmaxLoss, self).__init__(model, is_pairwise=True, reg_type=reg_type)
        self.loss = nn.CrossEntropyLoss()

    def lossFn(self, p_score, n_score):
        # Positive and negative classes.
        ones, zeros = torch.ones_like(p_score), torch.zeros_like(n_score)
        return self.loss(torch.concat((p_score, n_score), dim=1), torch.concat((ones, zeros), dim=1))
