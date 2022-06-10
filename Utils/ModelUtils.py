from Models.Analogy import Analogy
from Models.AprilE import AprilE
from Models.AttE import AttE
from Models.AttH import AttH
from Models.BoxE import BoxE
from Models.CombinE import CombinE
from Models.ComplEx import ComplEx
from Models.CrossE import CrossE
from Models.CyclE import CyclE
from Models.DistMult import DistMult
from Models.DensE import DensE
from Models.GCOTE import GCOTE
from Models.HAKE import HAKE
from Models.HARotatE import HARotatE
from Models.HolE import HolE
from Models.HyperKG import HyperKG
from Models.GTrans import GTrans
from Models.KG2E import KG2E
from Models.LineaRE import LineaRE
from Models.lppTransE import lppTransE
from Models.MAKR import MAKR
from Models.ManifoldE import ManifoldE
from Models.MDE import MDE
from Models.MRotatE import MRotatE
from Models.MuRE import MuRE
from Models.MuRP import MuRP
from Models.NagE import NagE
from Models.pRotatE import pRotatE
from Models.PairRE import PairRE
from Models.QuatDE import QuatDE
from Models.QuatE import QuatE
from Models.RatE import RatE
from Models.RESCAL import RESCAL
from Models.ReflectE import ReflectE
from Models.RotatE import RotatE
from Models.RotatE3D import RotatE3D
from Models.RotPro import RotPro
from Models.SE import SE
from Models.SimplE import SimplE
from Models.STransE import STransE
from Models.StructurE import StructurE
from Models.TransA import TransA
from Models.TransAt import TransAt
from Models.TransD import TransD
from Models.TransDR import TransDR
from Models.TransE import TransE
from Models.TransEDT import TransEDT
from Models.TransEdge import TransEdge
from Models.TransEFT import TransEFT
from Models.TransERS import TransERS
from Models.TransF import TransF
from Models.TransGate import TransGate
from Models.TransH import TransH
from Models.TransM import TransM
from Models.TransMS import TransMS
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
    if model_name == "transd" or model_name == "transr" or model_name == "tucker" or model_name == "transsparse" or \
            model_name == "transdr":
        kwargs.update({"dim_e":params["dime"], "dim_r":params["dimr"]})
    else:
        kwargs.update({"dim": params["dim"]})
    if model_name == "transe" or model_name == "toruse" or model_name == "stranse" or model_name == "transsparse" or \
        model_name == "boxe" or model_name == "makr" or model_name == "transms" or model_name == "transers" or \
            model_name == "lpptranse" or model_name == "transeft" or model_name == "transm" or model_name == "mde" or \
            model_name == "combine" or model_name == "transgate" or model_name == "transat" or \
            model_name == "aprile" or model_name == "reflecte" or model_name == "structure" or \
            model_name == "transedt" or model_name == "transedge" or model_name == "harotate" or \
            model_name == "manifolde":
        kwargs.update({"norm": params["pnorm"]})
    if model_name == "transsparse" or model_name == "transm":
        kwargs.update({"pred_count": params["pred_count"], "pred_loc_count": params["pred_loc_count"]})
    if model_name == "transsparse":
        kwargs.update({"type": params["sparse_type"]})
    if model_name == "gcote" or model_name == "gtrans":
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
    elif model_name == "lpptranse":
        m = lppTransE(**kwargs)
    elif model_name == "transers":
        m = TransERS(**kwargs)
    elif model_name == "transms":
        m = TransMS(**kwargs)
    elif model_name == "makr":
        m = MAKR(**kwargs)
    elif model_name == "aprile":
        m = AprilE(**kwargs)
    elif model_name == "rate":
        m = RatE(**kwargs)
    elif model_name == "hyperkg":
        m = HyperKG(**kwargs)
    elif model_name == "mrotate":
        m = MRotatE(**kwargs)
    elif model_name == "harotate":
        m = HARotatE(**kwargs)
    elif model_name == "pairre":
        m = PairRE(**kwargs)
    elif model_name == "cycle":
        m = CyclE(**kwargs)
    elif model_name == "reflecte":
        m = ReflectE(**kwargs)
    elif model_name == "structure":
        m = StructurE(**kwargs)
    elif model_name == "gtrans":
        m = GTrans(**kwargs)
    elif model_name == "transedt":
        m = TransEDT(**kwargs)
    elif model_name == "transdr":
        m = TransDR(**kwargs)
    elif model_name == "transedge":
        m = TransEdge(**kwargs)
    elif model_name == "se":
        m = SE(**kwargs)
    elif model_name == "amie":
        m = Models.Amie()

    return m


    # TODO: https://github.com/xinguoxia/KGE
    # ConE (https://arxiv.org/pdf/2110.14923v2.pdf) works with hierarchies that can be pre-computed, see Appendix H.
    # ITransF: https://aclanthology.org/P17-1088.pdf
    # DihEdral: https://aclanthology.org/P19-1026.pdf
    # TransC (requires instanceOf and subclassOf triples): https://aclanthology.org/D18-1222.pdf
    # KEC (requires concepts): https://www.sciencedirect.com/science/article/pii/S0950705118304945
    # ConnectE (requires type info): https://www.sciencedirect.com/science/article/abs/pii/S0950705120301921
    # TransRHS (requires subrelationOf): https://doi.org/10.24963/ijcai.2020/413
    # SSE (requires concepts): https://www.aclweb.org/anthology/P15-1009/

    # Neural networks
    # Hyper: https://arxiv.org/pdf/1808.07018v5.pdf
    # CapsE: https://arxiv.org/pdf/1808.04122v3.pdf