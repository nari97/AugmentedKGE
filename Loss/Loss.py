import torch
import torch.nn as nn


class Loss(nn.Module):
    def __init__(self, model, is_pairwise=False, reg_type='L2'):
        super(Loss, self).__init__()
        self.model = model
        self.nr = model.hyperparameters["nr"]
        self.wc = model.hyperparameters["weight_constraints"]
        self.lmbda = model.hyperparameters["lmbda"]
        # This tells us whether or not we need to split positives and negatives.
        self.is_pairwise = is_pairwise
        self.reg_type = reg_type

    def _get_positive_score(self, score, y):
        # Score is a 1D array, converting it to a 2D array
        positive_score = score[y == 1]
        positive_score = positive_score.view(positive_score.shape[0], 1)
        return positive_score

    def _get_negative_score(self, score, y):
        # Score is a 1D array, convert to array of size (batch_size, negative_rate)
        negative_score = score[y == -1].view(-1, self.nr)
        return negative_score

    def forward(self, data):
        score = self.model(data)
        if self.is_pairwise:
            p_score = self._get_positive_score(score, data["batch_y"])
            n_score = self._get_negative_score(score, data["batch_y"])
            loss_res = self.lossFn(p_score, n_score)
        else:
            loss_res = self.lossFn(score, data["batch_y"])

        # Apply constraints.
        constraints = torch.tensor([self.model.constraints(data)], device=score.device)
        # Apply regularization.
        regularization = torch.tensor([self.model.regularization(data, reg_type=self.reg_type)], device=score.device)

        return loss_res + self.wc * constraints + self.lmbda * regularization