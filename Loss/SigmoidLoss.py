import torch
import torch.nn as nn
import torch.nn.functional as F
from .Loss import Loss

class SigmoidLoss(Loss):

    def __init__(self, model):
        super(SigmoidLoss, self).__init__(model)
        self.criterion = nn.LogSigmoid()

    def lossFn(self, p_score, n_score):
        return -(self.criterion(p_score).mean() + self.criterion(-n_score).mean())/2

