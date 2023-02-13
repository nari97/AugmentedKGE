from Loss.MarginLoss import MarginLoss
from Loss.BCELoss import BCELoss
from Loss.SoftMarginLoss import SoftMarginLoss
from Loss.LimitLoss import LimitLoss
from Loss.MarginLimitLoss import MarginLimitLoss
import torch.nn as nn


def getLoss(model, loss_str=None, margin=0, other_margin=0, reg_type='L2'):
    """
        Gets the loss function based on model

        Args:
            model (Model): Model to get wrap with loss function.
            gamma (float): Margin for loss functions.
            

        Returns:
            loss: Loss function selected according to model
    """

    kwargs = {"model": model, "reg_type": reg_type, "margin" : margin}

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
        print ('Loss: Soft Margin Loss with margin:', margin)
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
        kwargs.update({"margin_p": margin, "margin_n": other_margin})
        print ('Loss: Limit-based with positive margin:', margin, ' and negative margin:', other_margin)
        loss = LimitLoss(**kwargs)
    elif loss_str is 'margin_limit':
        kwargs.pop('margin')
        kwargs.update({"margin_r": margin, "margin_s": other_margin})
        print ('Loss: Margin-based Ranking with margin:', margin, ' and limit-based with margin:', other_margin)
        loss = MarginLimitLoss(**kwargs)
    else:
        pass

    return loss
