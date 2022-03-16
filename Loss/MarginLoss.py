import torch
from torch.functional import Tensor
import torch.nn as nn
import torch.nn.functional as F
from .Loss import Loss
from Utils.utils import to_var

class MarginLoss(Loss):

    def __init__(self, model, margin=6.0):
        super(MarginLoss, self).__init__(model)
        self.loss = nn.MarginRankingLoss(margin)


    def lossFn(self, p_score, n_score):
        t = torch.ones((len(p_score), 1))
        ones = Tensor(t)
        if p_score.is_cuda:
            ones = ones.cuda()
        return self.loss(p_score, n_score, ones)
