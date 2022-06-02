from Models.Analogy import Analogy
from Models.AttE import AttE
from Models.AttH import AttH
from Models.BoxE import BoxE
from Models.CombinE import CombinE
from Models.ComplEx import ComplEx
from Models.CrossE import CrossE
from Models.DistMult import DistMult
from Models.DensE import DensE
from Models.GCOTE import GCOTE
from Models.HAKE import HAKE
from Models.HolE import HolE
from Models.KG2E import KG2E
from Models.LineaRE import LineaRE
from Models.ManifoldE import ManifoldE
from Models.MDE import MDE
from Models.MuRE import MuRE
from Models.MuRP import MuRP
from Models.NagE import NagE
from Models.pRotatE import pRotatE
from Models.QuatDE import QuatDE
from Models.QuatE import QuatE
from Models.RESCAL import RESCAL
from Models.RotatE import RotatE
from Models.RotatE3D import RotatE3D
from Models.RotPro import RotPro
from Models.SimplE import SimplE
from Models.STransE import STransE
from Models.TransA import TransA
from Models.TransAt import TransAt
from Models.TransD import TransD
from Models.TransE import TransE
from Models.TransEFT import TransEFT
from Models.TransF import TransF
from Models.TransGate import TransGate
from Models.TransH import TransH
from Models.TransM import TransM
from Models.TransR import TransR
from Models.TransSparse import TransSparse
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
    if model_name == "transd" or model_name == "transr" or model_name == "tucker" or model_name == "transsparse":
        kwargs.update({"dim_e":params["dime"], "dim_r":params["dimr"]})
    else:
        kwargs.update({"dim": params["dim"]})
    if model_name == "transe" or model_name == "toruse" or model_name == "stranse" or model_name == "transsparse" or \
        model_name == "boxe":
        kwargs.update({"norm": params["pnorm"]})
    if model_name == "transsparse" or model_name == "transm":
        kwargs.update({"pred_count": params["pred_count"], "pred_loc_count": params["pred_loc_count"]})
    if model_name == "transsparse":
        kwargs.update({"type": params["sparse_type"]})
    if model_name == "gcote":
        kwargs.update({"head_context":params["head_context"], "tail_context":params["tail_context"]})

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
    elif model_name == "transsparse":
        m = TransSparse(**kwargs)
    elif model_name == "manifolde":
        m = ManifoldE(**kwargs)
    elif model_name == "stranse":
        m = STransE(**kwargs)
    elif model_name == "crosse":
        m = CrossE(**kwargs)
    elif model_name == "hake":
        m = HAKE(**kwargs)
    elif model_name == "transf":
        m = TransF(**kwargs)
    elif model_name == "transa":
        m = TransA(**kwargs)
    elif model_name == "rescal":
        m = RESCAL(**kwargs)
    elif model_name == "murp":
        m = MuRP(**kwargs)
    elif model_name == "mure":
        m = MuRE(**kwargs)
    elif model_name == "atte":
        m = AttE(**kwargs)
    elif model_name == "atth":
        m = AttH(**kwargs)
    elif model_name == "transat":
        m = TransAt(**kwargs)
    elif model_name == "dense":
        m = DensE(**kwargs)
    elif model_name == "nage":
        m = NagE(**kwargs)
    elif model_name == "rotate3d":
        m = RotatE3D(**kwargs)
    elif model_name == "lineare":
        m = LineaRE(**kwargs)
    elif model_name == "gcote":
        m = GCOTE(**kwargs)
    elif model_name == "quatde":
        m = QuatDE(**kwargs)
    elif model_name == "boxe":
        m = BoxE(**kwargs)
    elif model_name == "rotpro":
        m = RotPro(**kwargs)
    elif model_name == "mde":
        m = MDE(**kwargs)
    elif model_name == "transm":
        m = TransM(**kwargs)
    elif model_name == "protate":
        m = pRotatE(**kwargs)
    elif model_name == "transgate":
        m = TransGate(**kwargs)
    elif model_name == "combine":
        m = CombinE(**kwargs)
    elif model_name == "kg2e":
        m = KG2E(**kwargs)
    elif model_name == "transeft":
        m = TransEFT(**kwargs)
    elif model_name == "amie":
        m = Models.Amie()

    return m


    # TODO: https://github.com/xinguoxia/KGE
    # ConE (https://arxiv.org/pdf/2110.14923v2.pdf) works with hierarchies that can be pre-computed, see Appendix H.

    # Neural networks
    # Hyper: https://arxiv.org/pdf/1808.07018v5.pdf
    # CapsE: https://arxiv.org/pdf/1808.04122v3.pdf