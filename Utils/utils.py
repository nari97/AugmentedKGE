import torch
import Loss
import pickle
from Train.Evaluator import RankCollector
from Models.ComplEx import ComplEx
from Models.DistMult import DistMult
from Models.HolE import HolE
from Models.RotatE import RotatE
from Models.SimplE import SimplE
from Models.TransD import TransD
from Models.TransE import TransE
from Models.TransH import TransH
from Loss.MarginLoss import MarginLoss
from Loss.MarginSigmoidLoss import MarginSigmoidLoss
from Loss.NegativeSamplingLoss import NegativeSamplingLoss
from Loss.SigmoidLoss import SigmoidLoss
from Loss.SoftplusLoss import SoftplusLoss



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
        loss = MarginLoss(margin=gamma)
        print ('Loss : Margin Loss')
    elif model_name == 'hole' or model_name == 'distmult':
        loss = MarginSigmoidLoss(margin = gamma)
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
            m = TransE(
                ent_total = params["ent_total"],
                rel_total = params["rel_total"],
                dims = params["dim"],
                norm = params["pnorm"],
                inner_norm = params["inner_norm"])
        elif model_name == "transh":
            m = TransH(
                ent_total = params["ent_total"],
                rel_total = params["rel_total"],
                dims=params["dim"],
                norm=params["pnorm"],
                inner_norm = params["inner_norm"])
        elif model_name == "transd":
            m = TransD(
                ent_total = params["ent_total"],
                rel_total = params["rel_total"],
                dim_e=params["dime"],
                dim_r=params["dimr"],
                norm = params["pnorm"],
                inner_norm = params["inner_norm"])
        elif model_name == "distmult":
            m = DistMult(
                ent_total = params["ent_total"],
                rel_total = params["rel_total"],
                dims = params["dim"],
                inner_norm = params["inner_norm"])
        elif model_name == "complex":
            m = ComplEx(
                ent_total = params["ent_total"],
                rel_total = params["rel_total"],
                dims = params["dim"],
                inner_norm = params["inner_norm"])
        elif model_name == "hole":
            m = HolE(
                ent_total = params["ent_total"],
                rel_total = params["rel_total"],
                dims = params["dim"],
                inner_norm = params["inner_norm"])
        elif model_name == "simple":
            m = SimplE(
                ent_total = params["ent_total"],
                rel_total = params["rel_total"],
                dims = params["dim"])
        elif model_name == "rotate":
            m = RotatE(
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

