import torch
import torch.nn as nn
class Loss(nn.Module):
    def __init__(self, model):
        super(Loss, self).__init__()
        self.model = model
        self.nr = model.hyperparameters["nr"]
        
    #Moved Strategy inside loss

    def _get_positive_score(self, score, y):

        #Score is a 1D array, converting it to a 2D array
        positive_score = score[y == 1]
        positive_score = positive_score.view(positive_score.shape[0], 1)

        return positive_score

    def _get_negative_score(self, score, y):
        
        #Score is a 1D array, convert to array of size (batch_size, negative_rate)
        negative_score = score[y == -1].view(-1,self.nr)

        return negative_score

    def forward(self, data):

        score = self.model(data)
        p_score = self._get_positive_score(score, data["batch_y"])
        n_score = self._get_negative_score(score, data["batch_y"])
        loss_res = self.lossFn(p_score, n_score)
        return loss_res


    