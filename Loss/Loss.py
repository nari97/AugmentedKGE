import torch
import torch.nn as nn


class Loss(nn.Module):
    def __init__(self, model, is_pairwise=False, reg_type='L2'):
        super(Loss, self).__init__()
        self.model = model
        self.wc = model.hyperparameters["weight_constraints"]
        self.lmbda = model.hyperparameters["lmbda"]
        # This tells us whether or not we need to split positives and negatives.
        self.is_pairwise = is_pairwise
        self.reg_type = reg_type

        self.hyperparameters_used = {"weight_constraints": model.hyperparameters["weight_constraints"],
                                     "lmbda": model.hyperparameters["lmbda"]}

    # These are the hyperparameters actually used by the loss function.
    def set_hyperparameters_used(self, hyperparameters_used):
        self.hyperparameters_used.update(hyperparameters_used)

    def get_hyperparameters_used(self):
        return self.hyperparameters_used

    def _get_positive_score(self, score, y):
        # Score is a 1D array, converting it to a 2D array
        positive_score = score[y == 1]
        positive_score = positive_score.view(positive_score.shape[0], 1)
        return positive_score

    def _get_negative_score(self, score, y, pos_shape):
        # Score is a 1D array, convert to array of size (batch_size, negative_rate)
        negative_score = score[y == -1].view(pos_shape, -1)
        return negative_score

    def forward(self, data):
        score = self.model(data)
        if self.is_pairwise:
            p_score = self._get_positive_score(score, data["batch_y"])
            # Use the shape of the positive scores to determine the shape of the negative scores.
            n_score = self._get_negative_score(score, data["batch_y"], p_score.shape[0])
            loss_res = self.lossFn(p_score, n_score)
        else:
            loss_res = self.lossFn(score, data["batch_y"])

        # Apply constraints.
        constraints = self.model.constraints(data)
        # Apply regularization.
        regularization = self.model.regularization(data, reg_type=self.reg_type)

        return loss_res + self.wc * constraints + self.lmbda * regularization
