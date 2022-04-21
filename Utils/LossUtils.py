from Loss.MarginLoss import MarginLoss
from Loss.BCELoss import BCELoss
from Loss.SoftMarginLoss import SoftMarginLoss
import torch.nn as nn


def getLoss(model, gamma=0, reg_type='L2'):
    """
        Gets the loss function based on model

        Args:
            model (Model): Model to get wrap with loss function.
            gamma (float): Margin for loss functions.
            

        Returns:
            loss: Loss function selected according to model
    """

    model_name = model.model_name
    if model_name == "transe" or model_name == "transh" or model_name == "transd" or model_name == "transr" or \
            model_name == "toruse":
        loss = MarginLoss(model=model, margin=gamma, reg_type=reg_type)
        print ('Loss : Margin Loss')
    elif model_name == 'distmult' or model_name == 'hole':
        loss = MarginLoss(model=model, margin=gamma, criterion=nn.Sigmoid(), reg_type=reg_type)
        print ('Loss : Margin Sigmoid Loss')
    elif model_name == "rotate":
        loss = SoftMarginLoss(model=model, margin=gamma, reg_type=reg_type)
        print ('Loss: Soft Margin Loss with gamma: ', gamma)
    elif model_name == "analogy" or model_name == "quate" or model_name == "simple":
        loss = SoftMarginLoss(model=model, reg_type=reg_type)
        print('Loss: Soft Margin Loss')
    elif model_name == "complex":
        print ('Loss : BCE Loss with logits')
        loss = BCELoss(model=model, with_logits=True, reg_type=reg_type)
    else:
        pass

    return loss
