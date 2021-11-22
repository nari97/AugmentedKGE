import torch
import torch.nn as nn
class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()

    def _get_positive_score(self, score, y):
        # positive_score = score[:self.batch_size]
        # positive_score = positive_score.view(-1, self.batch_size).permute(1, 0)

        positive_score = score[y == 1]
        return positive_score

    def _get_negative_score(self, score, y):
        # negative_score = score[self.batch_size:]
        # negative_score = negative_score.view(-1, self.batch_size).permute(1, 0)
        negative_score = score[y == -1].view(-1,self.batch_size).permute(1,0)
        return negative_score

    def forward(self, data):
        self.lossFn(None, None)
        score = self.model(data)
        p_score = self._get_positive_score(score)
        n_score = self._get_negative_score(score)


        loss_res = self.lossFn(p_score, n_score)
        return loss_res


    