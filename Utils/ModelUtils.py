from Models.ComplEx import ComplEx
from Models.DistMult import DistMult
from Models.HolE import HolE
from Models.RotatE import RotatE
from Models.SimplE import SimplE
from Models.TransD import TransD
from Models.TransE import TransE
from Models.TransH import TransH

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
                inner_norm = params["inner_norm"])
        elif model_name == "transd":
            m = TransD(
                ent_total = params["ent_total"],
                rel_total = params["rel_total"],
                dim_e=params["dime"],
                dim_r=params["dimr"],
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