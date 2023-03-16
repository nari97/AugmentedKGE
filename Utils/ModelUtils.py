from Models.Analogy import Analogy
from Models.AprilE import AprilE
from Models.AttE import AttE
from Models.AttH import AttH
from Models.BoxE import BoxE
from Models.CombinE import CombinE
from Models.ComplEx import ComplEx
from Models.ConvE import ConvE
from Models.CrossE import CrossE
from Models.CyclE import CyclE
from Models.DistMult import DistMult
from Models.DensE import DensE
from Models.GCOTE import GCOTE
from Models.GTrans import GTrans
from Models.HAKE import HAKE
from Models.HARotatE import HARotatE
from Models.HolE import HolE
from Models.HypER import HypER
from Models.HyperKG import HyperKG
from Models.ITransF import ITransF
from Models.KG2E import KG2E
from Models.LineaRE import LineaRE
from Models.lppTransD import lppTransD
from Models.lppTransE import lppTransE
from Models.lppTransR import lppTransR
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
from Models.SEEK import SEEK
from Models.SimplE import SimplE
from Models.STransE import STransE
from Models.SpacE import SpacE
from Models.StructurE import StructurE
from Models.Trans4E import Trans4E
from Models.TransA import TransA
from Models.TransAt import TransAt
from Models.TransD import TransD
from Models.TransComplEx import TransComplEx
from Models.TransDR import TransDR
from Models.TransE import TransE
from Models.TransEDT import TransEDT
from Models.TransEdge import TransEdge
from Models.TransEFT import TransEFT
from Models.TransERS import TransERS
from Models.TransGate import TransGate
from Models.TransH import TransH
from Models.TransHFT import TransHFT
from Models.TransM import TransM
from Models.TransMS import TransMS
from Models.TransR import TransR
from Models.TransRDT import TransRDT
from Models.TransRFT import TransRFT
from Models.TransSparse import TransSparse
from Models.TransSparseDT import TransSparseDT
from Models.TorusE import TorusE
from Models.TuckER import TuckER


def getModel(model_name, params, other_params=None):
    """
    Gets the model object

    Args:
        model_name (str): Model name
        params (dict): Dictionary containing the hyperparameters of the experiment
        other_params (dict): these are special hyperparameters that will go directly to the construction of the model.

    Returns:
        m (Model): Instantiated model object
    """

    kwargs = {"ent_total":params["ent_total"], "rel_total":params["rel_total"]}

    # Dealing with variants. The convention is that if a model name contains _ (underscore), it means it is a variant.
    if '_' in model_name:
        model_name, variant = model_name.split('_')
        kwargs.update({'variant': variant})

    if model_name == "transd" or model_name == "transr" or model_name == "tucker" or model_name == "transsparse" or \
        model_name == "hyper" or model_name == "lpptransr" or model_name == "lpptransd" or model_name == "transrdt" or \
            model_name == "transsparsedt" or model_name == "transrft":
        kwargs.update({"dim_e":params["dime"], "dim_r":params["dimr"]})
    else:
        kwargs.update({"dim": params["dim"]})
    if model_name == "transe" or model_name == "stranse" or model_name == "transsparse" or model_name == "rotpro" or \
        model_name == "boxe" or model_name == "makr" or model_name == "transms" or model_name == "transers" or \
            model_name == "lpptranse" or model_name == "cycle" or model_name == "combine" or \
            model_name == "transgate" or  model_name == "aprile" or model_name == "combine" or \
            model_name == "transgate" or  model_name == "aprile" or model_name == "reflecte" or \
            model_name == "structure" or model_name == "transedt" or model_name == "harotate" or \
            model_name == "manifolde" or model_name == "transm" or model_name == "transat" or \
            model_name == "transrdt" or model_name == "transsparsedt" or model_name == "space" or \
            model_name == "itransf" or model_name == "transcomplex" or model_name == "trans4e":
        kwargs.update({"norm": params["pnorm"]})
    if model_name == "transsparse" or model_name == "transm" or model_name == "transsparsedt":
        kwargs.update({"pred_count": params["pred_count"], "pred_loc_count": params["pred_loc_count"]})
    if model_name == "gcote" or model_name == "gtrans":
        kwargs.update({"head_context":params["head_context"], "tail_context":params["tail_context"]})

    if other_params is not None:
        kwargs.update(other_params)

    # TODO Can we do this dynamically?
    # TODO Careful with models implemented like DistMult using tanh (extra parameters). KG2E also has extra.
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
    elif model_name == "conve":
        m = ConvE(**kwargs)
    elif model_name == "hyper":
        m = HypER(**kwargs)
    elif model_name == "lpptransr":
        m = lppTransR(**kwargs)
    elif model_name == "lpptransd":
        m = lppTransD(**kwargs)
    elif model_name == "transrdt":
        m = TransRDT(**kwargs)
    elif model_name == "transsparsedt":
        m = TransSparseDT(**kwargs)
    elif model_name == "transhft":
        m = TransHFT(**kwargs)
    elif model_name == "transrft":
        m = TransRFT(**kwargs)
    elif model_name == "space":
        m = SpacE(**kwargs)
    elif model_name == "seek":
        m = SEEK(**kwargs)
    elif model_name == "itransf":
        m = ITransF(**kwargs)
    elif model_name == "transcomplex":
        m = TransComplEx(**kwargs)
    elif model_name == "trans4e":
        m = Trans4E(**kwargs)
    elif model_name == "amie":
        m = Models.Amie()

    return m


    # TODO: https://github.com/xinguoxia/KGE
    # TODO: https://github.com/LIANGKE23/Awesome-Knowledge-Graph-Reasoning
    # ConE (https://arxiv.org/pdf/2110.14923v2.pdf) works with hierarchies that can be pre-computed, see Appendix H.
    # DihEdral: https://aclanthology.org/P19-1026.pdf
    # TransC (requires instanceOf and subclassOf triples): https://aclanthology.org/D18-1222.pdf
    # KEC (requires concepts): https://www.sciencedirect.com/science/article/pii/S0950705118304945
    # ConnectE (requires type info): https://www.sciencedirect.com/science/article/abs/pii/S0950705120301921
    # TransRHS (requires subrelationOf): https://doi.org/10.24963/ijcai.2020/413
    # SSE (requires concepts): https://www.aclweb.org/anthology/P15-1009/
    # NTN: Socher, Richard, Chen, Danqi, Manning, Christopher D., and Ng, Andrew Y. Reasoning with neural
    #           tensor networks for knowledge base completion. In NIPS, 2013.
    # https://www.ijcai.org/Abstract/16/421 (requires types)
    # RodE: https://ieeexplore.ieee.org/document/9240950
    # 5*E: https://ojs.aaai.org/index.php/AAAI/article/view/17095
    # ODE: https://aclanthology.org/2021.emnlp-main.750/
    # Flexible: https://dl.acm.org/doi/10.5555/3032027.3032102
    # AEM: https://ieeexplore.ieee.org/document/8545570
    # TRPE: https://www.sciencedirect.com/science/article/pii/S092523122200889X
    # GIE: https://ojs.aaai.org/index.php/AAAI/article/view/20491
    # MEI: Tran, Hung Nghiep; Takasu, Atsuhiro (2020). "Multi-Partition Embedding Interaction with Block Term
    #           Format for Knowledge Graph Completion". ECAI.
    # MEIM: Tran, Hung-Nghiep; Takasu, Atsuhiro (2022). "MEIM: Multi-partition Embedding Interaction Beyond Block
    #           Term Format for Efficient and Expressive Link Prediction". IJCAI.
    # TransComplex: https://arxiv.org/pdf/1909.00519.pdf
    # GeomE: https://aclanthology.org/2020.coling-main.46.pdf
    # RUGE: https://arxiv.org/abs/1711.11231
    # ALMP: https://link.springer.com/chapter/10.1007/978-3-031-10983-6_50
    # Tatec: https://jair.org/index.php/jair/article/view/10993
    # HypE: https://www.ijcai.org/proceedings/2020/303
    # RTransE: Alberto Garc´ıa-Dur´an, Antoine Bordes, and Nicolas Usunier. 2015. Composing Relationships with
    #   Translations.
    # PTransE: Yankai Lin, Zhiyuan Liu, Huanbo Luan, Maosong Sun, Siwei Rao, and Song Liu. 2015a. Modeling Relation
    #   Paths for Representation Learning of Knowledge Bases.
    # TimE: https://www.sciencedirect.com/science/article/abs/pii/S0950705120306936
    # ProtoE: https://www.mdpi.com/2078-2489/13/8/354
    # GTransE: https://link.springer.com/chapter/10.1007/978-3-030-39878-1_16
    # TransERS propose modifications over all trans* models.
    # TransMVG: https://link.springer.com/chapter/10.1007/978-3-030-62005-9_21
    # RGKE: https://link.springer.com/chapter/10.1007/978-3-030-16142-2_37
    # KALE: https://aclanthology.org/D16-1019.pdf
    # KGE-CL: Zhiping Luo, Wentao Xu, Weiqing Liu, Jiang Bian, Jian Yin, Tie-Yan Liu: KGE-CL: Contrastive Learning of
    #   Tensor Decomposition Based Knowledge Graph Embeddings. COLING 2022: 2598-2607
    # RefH: Ines Chami, Adva Wolf, Da-Cheng Juan, Frederic Sala, Sujith Ravi, and Christopher Ré. 2020. Low-Dimensional
    #   Hyperbolic Knowledge Graph Embeddings.ACL, 6901–6914.
    # UltraE: https://dl.acm.org/doi/10.1145/3534678.3539333
    # https://ieeexplore.ieee.org/document/9533372 (several models partially trained and combined)







    # Neural networks
    # CapsE: https://arxiv.org/pdf/1808.04122v3.pdf
    # CapsE: https://aclanthology.org/N19-1226/
    # MDE has a MDENN version.
    # ConKB: https://aclanthology.org/N18-2053/
    # WGE: https://arxiv.org/abs/2112.09231
    # https://www.sciencedirect.com/science/article/abs/pii/S095070512200870X
    # https://www.sciencedirect.com/science/article/abs/pii/S0950705121004500
    # https://aclanthology.org/2020.emnlp-main.460/
    # https://arxiv.org/abs/2205.12102
    # R-GCN: Schlichtkrull, M., Kipf, T.N., Bloem, P., van den Berg, R., Titov, I., Welling, M.: Modeling relational
    #   data with graph convolutional networks. ESWC 2018. 593–607.
    # CACL: Oh, B., Seo, S., Lee, K.: Knowledge graph completion by context-aware convolutional learning with multi-hop
    #   neighborhoods. In: CIKM, pp. 257–266 (2018).
    # ProjE: Shi, B., Weninger, T.: ProjE: embedding projection for knowledge graph completion. In: AAAI, pp. 1236–1242
    #   (2017).
    # LogicENN: https://arxiv.org/pdf/1908.07141.pdf


    # ???
    # NoGE: https://dl.acm.org/doi/10.1145/3488560.3502183
    # https://arxiv.org/abs/1909.03821
    # https://www.hindawi.com/journals/sp/2020/7084958/
    # GAKE: https://aclanthology.org/C16-1062/ (even though it supports knowledge graphs, it is for regular graphs.)
    # TransG: https://aclanthology.org/P16-1219/ (talks about CRP and how to get M_r from it.)
