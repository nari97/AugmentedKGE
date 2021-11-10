import torch
import Loss
import pickle
from Train.Evaluator import RankCollector
import Models

def clamp_norm(input, p = 2, dim = -1, maxnorm = 1):
    """
        Computed Ln norm and clamps value of input to max

        Args:
            input (Tensor): Tensor containing vector of data to be clamped
            p (int) : L1 or L2 norm. Default: 2
            dim (int) : Dimension across which norm is to be calculated. Default: -1
            maxnorm (int) : Maximum value after which data is clamped. Default: 1

        Returns:
            ans: Tensor that has been normalised and has had values clamped
    """

    norm = torch.norm(input, p = p, dim = dim, keepdim=True)
    mask = (norm<maxnorm).long()
    ans = mask*input + (1-mask)*(input/torch.clamp_min(norm, 10e-8)*maxnorm)
    return ans

def normalize(input, p = 2, dim = -1):
    """
        Computes Ln norm

        Args:
            input (Tensor): Tensor containing vector of data to be normalized
            p (int) : L1 or L2 norm. Default: 2
            dim (int) : Dimension across which norm is to be calculated. Default: -1
            

        Returns:
            ans: Tensor that has been normalised
    """

    ans = torch.nn.functional.normalize(input, p = p, dim = dim)
    return ans

def getLoss(model_name, gamma = 0):
    """
        Gets the loss function based on model

        Args:
            model_name (str): Model name
            gamma (float) : Margin for losses
            

        Returns:
            loss: Loss function selected according to model_name
    """

    if model_name == "transe" or model_name == "transh" or model_name == "transd":
        loss = Loss.MarginLoss.MarginLoss(margin=gamma)
        print ('Loss : Margin Loss')
    elif model_name == 'hole' or model_name == 'distmult':
        loss = Loss.MarginSigmoidLoss.MarginSigmoidLoss(margin = gamma)
        print ('Loss : Margin Sigmoid Loss')
    elif model_name == "rotate":
        loss = Loss.NegativeSamplingLoss.NegativeSamplingLoss(margin = gamma)
        print ('Loss : Negative Sampling Loss')
    elif model_name == "analogy":
        print ('Loss : Sigmoid Loss')
        loss = Loss.SigmoidLoss.SigmoidLoss()
    else:
        print ('Loss : Softplus Loss')
        loss = Loss.SoftplusLoss.SoftplusLoss()

    return loss

def reportMetrics(folder, dataset, index, model_name, ranks, totals, parameters):
    """
        Reports metrics and saves it as ax file

        Args:
            folder (str): Base folder
            dataset (int) : Dataset chosen
            index (int) : Index of experiment
            model_name (str) : Name of model
            ranks (list) : List of all the ranks from the experiment
            totals (list) : List of all the totals from the experiment
            parameters (dict) : Dictionary containing the hyperparameters of the experiment
            

        Returns:
            
    """

    rc = RankCollector()
    rc.load(ranks, totals)

    # Report metric!!!!!
    result = {}
    result['trial_index'] = parameters['trial_index']
    result['mrh'] = rc.get_metric().get()
    result_file = folder + "Ax/" + model_name + "_" + str(dataset) + "_" + str(index) + ".result"
    with open(result_file, 'wb') as f:
        pickle.dump(result, f, pickle.HIGHEST_PROTOCOL)

def getModel(model_name, params):
        """
        Gets the model object

        Args:
            model_name (str): Model name
            params (dict) : Dictionary containing the hyperparameters of the experiment
            

        Returns:
            m (Model): Instantiated model object
        """

        m = None
        if model_name == "transe":
            m = Models.TransE.TransE(
                ent_total = params["ent_total"],
                rel_total = params["rel_total"],
                dims = params["dim"],
                norm = params["pnorm"],
                inner_norm = params["inner_norm"])
        elif model_name == "transh":
            m = Models.TransH.TransH(
                ent_total = params["ent_total"],
                rel_total = params["rel_total"],
                dims=params["dim"],
                norm=params["pnorm"],
                inner_norm = params["inner_norm"])
        elif model_name == "transd":
            m = Models.TransD.TransD(
                ent_total = params["ent_total"],
                rel_total = params["rel_total"],
                dim_e=params["dime"],
                dim_r=params["dimr"],
                norm = params["pnorm"],
                inner_norm = params["inner_norm"])
        elif model_name == "distmult":
            m = Models.DistMult.DistMult(
                ent_total = params["ent_total"],
                rel_total = params["rel_total"],
                dims = params["dim"],
                inner_norm = params["inner_norm"])
        elif model_name == "complex":
            m = Models.ComplEx.ComplEx(
                ent_total = params["ent_total"],
                rel_total = params["rel_total"],
                dims = params["dim"],
                inner_norm = params["inner_norm"])
        elif model_name == "hole":
            m = Models.HolE.HolE(
                ent_total = params["ent_total"],
                rel_total = params["rel_total"],
                dims = params["dim"],
                inner_norm = params["inner_norm"])
        elif model_name == "simple":
            m = Models.SimplE.SimplE(
                ent_total = params["ent_total"],
                rel_total = params["rel_total"],
                dims = params["dim"])
        elif model_name == "analogy":
            m = Models.Analogy.Analogy(
                ent_total = params["ent_total"],
                rel_total = params["rel_total"],
                dims = params["dim"],
                inner_norm = params["inner_norm"])
        elif model_name == "rotate":
            m = Models.RotatE.RotatE(
                ent_total = params["ent_total"],
                rel_total = params["rel_total"],
                dims = params["dim"],
                inner_norm = params["inner_norm"])
        elif model_name == "amie":
            m = Models.Amie()

        return m

def check_params(params):
    """
        Checks hyperparameters for validity

        Args:
            params (dict) : Dictionary containing the hyperparameters of the experiment
            

        Returns:
    """

    flag = False
    if "nbatches" not in params:
        print("Number of batches are missing")
        flag = True
    if "nr" not in params:
        print ("Negative rate is missing")
        flag = True
    if "lr" not in params:
        print ("Learning rate is missing")
        flag = True
    if "wd" not in params:
        print ("Weight decay is missing")
        flag = True
    if "m" not in params:
        print ("Momentum is missing")
        flag = True
    if "trial_index" not in params:
        print ("Trial index is missing")
        flag = True
    if "dim" not in params or "dim_e" not in params or "dim_r" not in params:
        print ("Dimensions for embeddings are missing")
        flag = True
    if "pnorm" not in params:
        print ("Learning rate is missing")
        flag = True
    if "gamma" not in params:
        print ("Gamma is missing")
        flag = True
    if "inner_norm" not in params:
        print ("Inner norm is missing")
        flag = True

    if flag:
        exit(0)

