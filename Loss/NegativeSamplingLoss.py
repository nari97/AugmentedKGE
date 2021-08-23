import torch
import torch.nn as nn
import torch.nn.functional as F
from .Loss import Loss

class NegativeSamplingLoss(Loss):

    def __init__(self, adv_temperature=None, margin=6.0):
        super(NegativeSamplingLoss, self).__init__()
        self.criterion = nn.LogSigmoid()
        self.margin = margin
        if adv_temperature != None:
            self.adv_temperature = nn.Parameter(torch.Tensor([adv_temperature]))
            self.adv_temperature.requires_grad = False
            self.adv_flag = True
        else:
            self.adv_flag = False

    def get_weights(self, n_score):
        return F.softmax(n_score * self.adv_temperature, dim=-1).detach()

    def forward(self, p_score, n_score):
        
        return -(self.criterion(self.margin - p_score).mean() + self.criterion(n_score - self.margin).mean())

    def predict(self, p_score, n_score):
        score = self.forward(p_score, n_score)
        return score.cpu().data.numpy()
