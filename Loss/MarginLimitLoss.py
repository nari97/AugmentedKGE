from .Loss import Loss
from .MarginLoss import MarginLoss
from .LimitLoss import LimitLoss


class MarginLimitLoss(Loss):

    def __init__(self, model, margin_r=1e-1, margin_s=1e-1, lmbda=.8, criterion=None, reg_type='L2'):
        super(MarginLimitLoss, self).__init__(model, is_pairwise=True, reg_type=reg_type)
        self.loss_r = MarginLoss(model, margin=margin_r, criterion=criterion, reg_type=reg_type)
        # We will not use the negatives.
        self.loss_s = LimitLoss(model, margin_p=margin_s, alpha=0, criterion=criterion, reg_type=reg_type)
        self.lmbda = lmbda
        self.criterion = criterion

    def lossFn(self, p_score, n_score):
        return self.loss_r.lossFn(p_score, n_score) + self.lmbda * self.loss_s.lossFn(p_score, n_score)
