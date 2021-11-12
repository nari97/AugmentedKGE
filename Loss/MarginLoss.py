import torch
from torch.functional import Tensor
import torch.nn as nn
import torch.nn.functional as F
from .Loss import Loss
from Utils.utils import to_var

class MarginLoss(Loss):

    def __init__(self, adv_temperature=None, margin=6.0, use_gpu = False):
        super(MarginLoss, self).__init__()
        self.margin = nn.Parameter(torch.Tensor([margin]))
        self.margin.requires_grad = False
        self.loss = nn.MarginRankingLoss(margin)

        self.use_gpu = use_gpu
        
        if adv_temperature != None:
            self.adv_temperature = nn.Parameter(torch.Tensor([adv_temperature]))
            self.adv_temperature.requires_grad = False
            self.adv_flag = True
        else:
            self.adv_flag = False

    def get_weights(self, n_score):
        return F.softmax(-n_score * self.adv_temperature, dim=-1).detach()

    def forward(self, p_score, n_score):
        #print (p_score.shape)
        if self.adv_flag:
            return (self.get_weights(n_score) * torch.max(p_score - n_score, -self.margin)).sum(
                dim=-1).mean() + self.margin
        else:
            t = torch.ones((len(p_score), 1))
            ones = Tensor(t)


            if p_score.is_cuda:
                ones = ones.cuda()

            #print (p_score.device, n_score.device, ones.device)
            return self.loss(p_score, n_score, ones)

    def predict(self, p_score, n_score):
        score = self.forward(p_score, n_score)
        return score.cpu().data.numpy()