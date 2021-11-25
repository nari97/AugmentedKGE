import torch
import torch.nn as nn
import torch.nn.functional as F
from .Loss import Loss

class NegativeSamplingLoss(Loss):

    def __init__(self,model, margin=6.0):
        super(NegativeSamplingLoss, self).__init__(model)
        self.criterion = nn.LogSigmoid()
        self.margin = margin

    def lossFn(self, p_score, n_score): 
        return -(self.criterion(self.margin - p_score).mean() + self.criterion(n_score - self.margin).mean())

