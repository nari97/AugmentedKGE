from Loss.MarginLoss import MarginLoss
from Loss.BCELoss import BCELoss
from Loss.SoftMarginLoss import SoftMarginLoss
from Loss.LimitLoss import LimitLoss
from Loss.MarginLimitLoss import MarginLimitLoss
import torch.nn as nn


def getLoss(model, loss_str=None, gamma=0, other_gamma=0, reg_type='L2'):
    """
        Gets the loss function based on model

        Args:
            model (Model): Model to get wrap with loss function.
            gamma (float): Margin for loss functions.
            

        Returns:
            loss: Loss function selected according to model
    """

    kwargs = {"model": model, "reg_type": reg_type, "margin" : gamma}

    if loss_str is None:
        loss_str = model.get_default_loss()

    if loss_str is 'margin':
        loss = MarginLoss(**kwargs)
        print ('Loss: Margin-based Ranking Loss')
    elif loss_str is 'margin_sigmoid':
        kwargs.update({"criterion": nn.Sigmoid()})
        loss = MarginLoss(**kwargs)
        print ('Loss: Margin-based Ranking Sigmoid Loss')
    elif loss_str is 'soft_margin':
        loss = SoftMarginLoss(**kwargs)
        print ('Loss: Soft Margin Loss with margin:', gamma)
    elif loss_str is 'soft':
        kwargs.pop('margin')
        loss = SoftMarginLoss(**kwargs)
        print('Loss: Soft Margin Loss')
    elif loss_str is 'bce':
        kwargs.pop('margin')
        kwargs.update({"with_logits": True})
        print ('Loss: BCE Loss with logits')
        loss = BCELoss(**kwargs)
    elif loss_str is 'limit':
        kwargs.pop('margin')
        kwargs.update({"margin_p": gamma, "margin_n": other_gamma})
        print ('Loss: Limit-based with positive margin:', gamma, ' and negative margin:', other_gamma)
        loss = LimitLoss(**kwargs)
    elif loss_str is 'margin_limit':
        kwargs.pop('margin')
        kwargs.update({"margin_r": gamma, "margin_s": other_gamma})
        print ('Loss: Margin-based Ranking with margin:', gamma, ' and limit-based with margin:', other_gamma)
        loss = MarginLimitLoss(**kwargs)
    else:
        pass

    return loss
