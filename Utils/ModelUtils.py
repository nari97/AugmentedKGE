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
from Models.TuckER import TuckER


def getModel(model_name, params):
    """
    Gets the model object

    Args:
        model_name (str): Model name
        params (dict) : Dictionary containing the hyperparameters of the experiment

    Returns:
        m (Model): Instantiated model object
    """

    kwargs = {"ent_total":params["ent_total"], "rel_total":params["rel_total"]}
    if model_name == "transd" or model_name == "transr" or model_name == "tucker":
        kwargs.update({"dim_e":params["dime"], "dim_r":params["dimr"]})
    else:
        kwargs.update({"dim": params["dim"]})
    if model_name == "transe" or model_name == "toruse":
        kwargs.update({"norm": params["pnorm"]})

    m = None
    if model_name == "transe":
        m = TransE(**kwargs)
    elif model_name == "toruse":
        m = TorusE(**kwargs)
    elif model_name == "transh":
        m = TransH(**kwargs)
    elif model_name == "transd":
        m = TransD(**kwargs)
    elif model_name == "transr":
        m = TransR(**kwargs)
    elif model_name == "tucker":
        m = TuckER(**kwargs)
    elif model_name == "distmult":
        m = DistMult(**kwargs)
    elif model_name == "complex":
        m = ComplEx(**kwargs)
    elif model_name == "hole":
        m = HolE(**kwargs)
    elif model_name == "simple":
        m = SimplE(**kwargs)
    elif model_name == "rotate":
        m = RotatE(**kwargs)
    elif model_name == "analogy":
        m = Analogy(**kwargs)
    elif model_name == "quate":
        m = QuatE(**kwargs)
    elif model_name == "amie":
        m = Models.Amie()

    return m


    # TODO
    # Multi-relational poincaré graph embeddings
    # Low-dimensional hyperbolic knowledge graph embeddings
    # Hyperbolic entailment cones for learning hierarchical embeddings
    # Learning hierarchy-aware knowledge graph embeddings for link prediction
    # Wen Zhang, Bibek Paudel, Wei Zhang, Abraham Bernstein, and Huajun Chen. 2019. Interaction embeddings for
    #   prediction and explanation in knowledge graphs. In Proceedings of the 12th ACM International Conference on Web
    #   Search and Data Mining
    # Check this for latest: https://www.mdpi.com/2076-3417/12/8/3935

    # ConE: https://arxiv.org/pdf/2110.14923v2.pdf, https://github.com/snap-stanford/ConE/
    # This is mostly dealing with hierarchies, so it is necessary to know hierarchies between predicates in advanced.

    # Neural networks
    # Hyper: https://arxiv.org/pdf/1808.07018v5.pdf
    # CapsE: https://arxiv.org/pdf/1808.04122v3.pdf