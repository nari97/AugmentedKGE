from Loss.MarginLoss import MarginLoss
from Loss.BCELoss import BCELoss
from Loss.SoftMarginLoss import SoftMarginLoss
from Loss.LimitLoss import LimitLoss
from Loss.MarginLimitLoss import MarginLimitLoss
from Loss.LogSoftmaxLoss import LogSoftmaxLoss
import torch.nn as nn


def getLoss(model, loss_str=None, margin=0, other_margin=0, neg_weight=1.0, reg_type='L2'):
    """
        Gets the loss function based on model

        Args:
            model (Model): Model to get wrap with loss function.
            margin (float): Margin for loss functions.
            

        Returns:
            loss: Loss function selected according to model
    """

    kwargs = {"model": model, "reg_type": reg_type, "margin": margin}

    if loss_str is None:
        loss_str = model.get_default_loss()

    if loss_str == 'margin':
        loss = MarginLoss(**kwargs)
        print ('Loss: Margin-based Ranking Loss')
    elif loss_str == 'margin_sigmoid':
        kwargs.update({"criterion": nn.Sigmoid()})
        loss = MarginLoss(**kwargs)
        print ('Loss: Margin-based Ranking Sigmoid Loss')
    elif loss_str == 'soft_margin':
        loss = SoftMarginLoss(**kwargs)
        print ('Loss: Soft Margin Loss with margin:', margin)
    elif loss_str == 'soft':
        kwargs.pop('margin')
        loss = SoftMarginLoss(**kwargs)
        print('Loss: Soft Margin Loss')
    elif loss_str == 'bce':
        kwargs.pop('margin')
        kwargs.update({"with_logits": True})
        print ('Loss: BCE Loss with logits')
        loss = BCELoss(**kwargs)
    elif loss_str == 'logsoftmax':
        kwargs.pop('margin')
        print ('Loss: Log-Softmax Loss')
        loss = LogSoftmaxLoss(**kwargs)
    elif loss_str == 'limit':
        kwargs.pop('margin')
        kwargs.update({"margin_p": margin, "margin_n": other_margin, "alpha": neg_weight})
        print ('Loss: Limit-based with positive margin:', margin, ' and negative margin:', other_margin)
        loss = LimitLoss(**kwargs)
    elif loss_str == 'margin_limit':
        kwargs.pop('margin')
        kwargs.update({"margin_r": margin, "margin_s": other_margin, "lmbda": neg_weight})
        print ('Loss: Margin-based Ranking with margin:', margin, ' and limit-based with margin:', other_margin)
        loss = MarginLimitLoss(**kwargs)
    else:
        pass

    loss.set_hyperparameters_used(kwargs)

    return loss
