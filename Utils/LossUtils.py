from Loss.MarginLoss import MarginLoss
from Loss.BCELoss import BCELoss
from Loss.SoftMarginLoss import SoftMarginLoss
import torch.nn as nn


def getLoss(model, loss_str=None, gamma=0, reg_type='L2'):
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
        print ('Loss : Margin Loss')
    elif loss_str is 'margin_sigmoid':
        kwargs.update({"criterion": nn.Sigmoid()})
        loss = MarginLoss(**kwargs)
        print ('Loss : Margin Sigmoid Loss')
    elif loss_str is 'soft_margin':
        loss = SoftMarginLoss(**kwargs)
        print ('Loss: Soft Margin Loss with gamma: ', gamma)
    elif loss_str is 'soft':
        kwargs.pop('margin')
        loss = SoftMarginLoss(**kwargs)
        print('Loss: Soft Margin Loss')
    elif loss_str is 'bce':
        kwargs.pop('margin')
        kwargs.update({"with_logits": True})
        print ('Loss : BCE Loss with logits')
        loss = BCELoss(**kwargs)
    else:
        pass

    return loss
