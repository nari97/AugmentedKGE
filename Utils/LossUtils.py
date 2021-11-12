from Loss.MarginLoss import MarginLoss
from Loss.MarginSigmoidLoss import MarginSigmoidLoss
from Loss.NegativeSamplingLoss import NegativeSamplingLoss
from Loss.SigmoidLoss import SigmoidLoss
from Loss.SoftplusLoss import SoftplusLoss

def getLoss(model_name, gamma = 0, use_gpu = False):
    """
        Gets the loss function based on model

        Args:
            model_name (str): Model name
            gamma (float) : Margin for losses
            

        Returns:
            loss: Loss function selected according to model_name
    """

    if model_name == "transe" or model_name == "transh" or model_name == "transd":
        loss = MarginLoss(margin=gamma, use_gpu = use_gpu)
        print ('Loss : Margin Loss')
    elif model_name == 'hole' or model_name == 'distmult':
        loss = MarginSigmoidLoss(margin = gamma, use_gpu = use_gpu)
        print ('Loss : Margin Sigmoid Loss')
    elif model_name == "rotate":
        loss = NegativeSamplingLoss(margin = gamma)
        print ('Loss : Negative Sampling Loss')
    elif model_name == "analogy":
        print ('Loss : Sigmoid Loss')
        loss = SigmoidLoss()
    else:
        print ('Loss : Softplus Loss')
        loss = SoftplusLoss()

    return loss