from Models.Analogy import Analogy
from Models.ComplEx import ComplEx
from Models.DistMult import DistMult
from Models.HolE import HolE
from Models.QuatE import QuatE
from Models.RotatE import RotatE
from Models.SimplE import SimplE
from Models.TransD import TransD
from Models.TransE import TransE
from Models.TransH import TransH
from Models.TransR import TransR
from Models.TorusE import TorusE


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
            ent_total=params["ent_total"],
            rel_total=params["rel_total"],
            dims=params["dim"],
            norm=params["pnorm"],)
    elif model_name == "toruse":
        m = TorusE(
            ent_total=params["ent_total"],
            rel_total=params["rel_total"],
            dims=params["dim"],
            norm=params["pnorm"],)
    elif model_name == "transh":
        m = TransH(
            ent_total=params["ent_total"],
            rel_total=params["rel_total"],
            dims=params["dim"],)
    elif model_name == "transd":
        m = TransD(
            ent_total=params["ent_total"],
            rel_total=params["rel_total"],
            dim_e=params["dime"],
            dim_r=params["dimr"],)
    elif model_name == "transr":
        m = TransR(
            ent_total=params["ent_total"],
            rel_total=params["rel_total"],
            dim_e=params["dime"],
            dim_r=params["dimr"],)
    elif model_name == "distmult":
        m = DistMult(
            ent_total=params["ent_total"],
            rel_total=params["rel_total"],
            dims=params["dim"],)
    elif model_name == "complex":
        m = ComplEx(
            ent_total=params["ent_total"],
            rel_total=params["rel_total"],
            dims=params["dim"],)
    elif model_name == "hole":
        m = HolE(
            ent_total=params["ent_total"],
            rel_total=params["rel_total"],
            dims=params["dim"],)
    elif model_name == "simple":
        m = SimplE(
            ent_total=params["ent_total"],
            rel_total=params["rel_total"],
            dims=params["dim"])
    elif model_name == "rotate":
        m = RotatE(
            ent_total=params["ent_total"],
            rel_total=params["rel_total"],
            dims=params["dim"],)
    elif model_name == "analogy":
        m = Analogy(
            ent_total=params["ent_total"],
            rel_total=params["rel_total"],
            dims=params["dim"],)
    elif model_name == "quate":
        m = QuatE(
            ent_total=params["ent_total"],
            rel_total=params["rel_total"],
            dims=params["dim"],)
    elif model_name == "amie":
        m = Models.Amie()

    return m


    # TODO
    # Tucker: Tucker: Tensor factorization for knowledge graph completion
    # Multi-relational poincaré graph embeddings
    # Low-dimensional hyperbolic knowledge graph embeddings
    # Hyperbolic entailment cones for learning hierarchical embeddings
    # Learning hierarchy-aware knowledge graph embeddings for link prediction

    # ConE: https://arxiv.org/pdf/2110.14923v2.pdf, https://github.com/snap-stanford/ConE/
    # This is mostly dealing with hierarchies, so it is necessary to know hierarchies between predicates in advanced.

    # Neural networks
    # Hyper: https://arxiv.org/pdf/1808.07018v5.pdf
    # CapsE: https://arxiv.org/pdf/1808.04122v3.pdf