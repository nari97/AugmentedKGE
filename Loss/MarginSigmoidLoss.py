import torch
import torch.nn as nn
import torch.nn.functional as F
from .Loss import Loss
from Utils.utils import to_var


class MarginSigmoidLoss(Loss):

    def __init__(self,model,  margin=6.0):
        super(MarginSigmoidLoss, self).__init__(model)
        self.sigmoid = nn.Sigmoid()
        self.loss = nn.MarginRankingLoss(margin)

    def lossFn(self, p_score, n_score):

        t = torch.ones((len(p_score), 1))
        ones = torch.Tensor(t)

        if p_score.is_cuda:
            ones = ones.cuda()

        return self.loss(self.sigmoid(p_score), self.sigmoid(n_score), ones)
