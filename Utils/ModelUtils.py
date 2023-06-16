from Models.Analogy import Analogy
from Models.AprilE import AprilE
from Models.AttH import AttH
from Models.BoxE import BoxE
from Models.CombinE import CombinE
from Models.ComplEx import ComplEx
from Models.ConvE import ConvE
from Models.CP import CP
from Models.CrossE import CrossE
from Models.CyclE import CyclE
from Models.DihEdral import DihEdral
from Models.DistMult import DistMult
from Models.DensE import DensE
from Models.DualE import DualE
from Models.FiveStarE import FiveStarE
from Models.GCOTE import GCOTE
from Models.GeomE import GeomE
from Models.GIE import GIE
from Models.GTrans import GTrans
from Models.HAKE import HAKE
from Models.HARotatE import HARotatE
from Models.HolE import HolE
from Models.HopfE import HopfE
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
from Models.MuRP import MuRP
from Models.NagE import NagE
from Models.pRotatE import pRotatE
from Models.PairRE import PairRE
from Models.ProjE import ProjE
from Models.QuatDE import QuatDE
from Models.QuatE import QuatE
from Models.RatE import RatE
from Models.RESCAL import RESCAL
from Models.ReflectE import ReflectE
from Models.RodE import RodE
from Models.RotatE import RotatE
from Models.RotatE3D import RotatE3D
from Models.RotateCT import RotateCT
from Models.RotPro import RotPro
from Models.SAttLE import SAttLE
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
from Models.TransHRS import TransHRS
from Models.TransM import TransM
from Models.TransMS import TransMS
from Models.TransMVG import TransMVG
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
            model_name == "aprile" or model_name == "reflecte" or model_name == "transmvg" or \
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
    elif model_name == "geome":
        m = GeomE(**kwargs)
    elif model_name == "dihedral":
        m = DihEdral(**kwargs)
    elif model_name == "rode":
        m = RodE(**kwargs)
    elif model_name == "cp":
        m = CP(**kwargs)
    elif model_name == "transhrs":
        m = TransHRS(**kwargs)
    elif model_name == "gie":
        m = GIE(**kwargs)
    elif model_name == "duale":
        m = DualE(**kwargs)
    elif model_name == "proje":
        m = ProjE(**kwargs)
    elif model_name == "transmvg":
        m = TransMVG(**kwargs)
    elif model_name == "sattle":
        m = SAttLE(**kwargs)
    elif model_name == "hopfe":
        m = HopfE(**kwargs)
    elif model_name == "rotatect":
        m = RotateCT(**kwargs)
    elif model_name == "fivestare":
        m = FiveStarE(**kwargs)
    elif model_name == "amie":
        m = Models.Amie()

    return m


    # TODO: https://github.com/xinguoxia/KGE
    # TODO: https://github.com/LIANGKE23/Awesome-Knowledge-Graph-Reasoning
    # TODO: https://www.mdpi.com/2079-9292/11/23/3866 (surveys software frameworks).
    # ConE (https://arxiv.org/pdf/2110.14923v2.pdf) works with hierarchies that can be pre-computed, see Appendix H.
    # TransC (requires instanceOf and subclassOf triples): https://aclanthology.org/D18-1222.pdf
    # KEC (requires concepts): https://www.sciencedirect.com/science/article/pii/S0950705118304945
    # ConnectE (requires type info): https://www.sciencedirect.com/science/article/abs/pii/S0950705120301921
    # TransRHS (requires subrelationOf): https://doi.org/10.24963/ijcai.2020/413
    # SSE (requires concepts): https://www.aclweb.org/anthology/P15-1009/
    # RUGE (requires logic rules): https://arxiv.org/abs/1711.11231
    # https://www.ijcai.org/Abstract/16/421 (requires types)
    # GTransE (it is for uncertain knowledge graphs): https://link.springer.com/chapter/10.1007/978-3-030-39878-1_16
    # HypE (hypergraphs): https://www.ijcai.org/proceedings/2020/303
    # Caps-OWKG (uses text descriptions): https://doi.org/10.1007/s13042-020-01259-4
    # TransO (requires ontology information): https://link.springer.com/article/10.1007/s11280-022-01016-3



    # ODE: https://aclanthology.org/2021.emnlp-main.750/
    # AEM: https://ieeexplore.ieee.org/document/8545570
    # TRPE: https://www.sciencedirect.com/science/article/pii/S092523122200889X
    # MEI: Tran, Hung Nghiep; Takasu, Atsuhiro (2020). "Multi-Partition Embedding Interaction with Block Term
    #           Format for Knowledge Graph Completion". ECAI.
    # MEIM: Tran, Hung-Nghiep; Takasu, Atsuhiro (2022). "MEIM: Multi-partition Embedding Interaction Beyond Block
    #           Term Format for Efficient and Expressive Link Prediction". IJCAI.
    # ALMP: https://link.springer.com/chapter/10.1007/978-3-031-10983-6_50
    # Tatec: https://jair.org/index.php/jair/article/view/10993
    # RTransE: Alberto Garc´ıa-Dur´an, Antoine Bordes, and Nicolas Usunier. 2015. Composing Relationships with
    #   Translations.
    # PTransE: Yankai Lin, Zhiyuan Liu, Huanbo Luan, Maosong Sun, Siwei Rao, and Song Liu. 2015a. Modeling Relation
    #   Paths for Representation Learning of Knowledge Bases.
    # TimE: https://www.sciencedirect.com/science/article/abs/pii/S0950705120306936
    # ProtoE: https://www.mdpi.com/2078-2489/13/8/354
    # RGKE: https://link.springer.com/chapter/10.1007/978-3-030-16142-2_37
    # KALE: https://aclanthology.org/D16-1019.pdf
    # UltraE: https://dl.acm.org/doi/10.1145/3534678.3539333
    # https://ieeexplore.ieee.org/document/9533372 (several models partially trained and combined)
    # TransESS: https://ieeexplore.ieee.org/document/9360502
    # CFAG: https://ojs.aaai.org/index.php/AAAI/article/view/20337
    # TransGH: https://link.springer.com/chapter/10.1007/978-3-319-93698-7_48
    # CKRL: Xie, R., Liu, Z., Lin, F., Lin, L.: Does William Shakespeare really write hamlet? Knowledge representation
    #   learning with confidence. AAAI, 2018.
    # SME: A. Bordes, X. Glorot, J. Weston, and Y. Bengio, ``Joint learning of words and meaning representations for
    #   open-text semantic parsing,'' in JMLR, 2012.
    # SME+: A. Bordes, X. Glorot, J. Weston, and Y. Bengio, ``A semantic matching energy function for learning with
    #   multi-relational data - Application to word-sense disambiguation,'' Mach. Learn., vol. 94, no. 2, 2014.
    # LFM: R. Jenatton, N. L. Roux, A. Bordes, and G. R. Obozinski, ``A latent factor model for highly multi-relational
    #   data,'' in NIPS, 2012,.
    # BiMult: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8353191
    # SimEER: https://downloads.hindawi.com/journals/sp/2018/6325635.pdf
    # BiTransE: https://ieeexplore.ieee.org/document/9754641
    # DyHHE: Dingyang Duan, Daren Zha, Xiao Yang, Xiaobo Guo: Dynamic Heterogeneous Information Network Embedding in
    #   Hyperbolic Space. SEKE 2022: 281-286.
    # Wenying Feng, Daren Zha, Xiaobo Guo, Yao Dong, Yuanye He: Representing Knowledge Graphs with Gaussian Mixture
    #   Embedding. KSEM 2021: 166-178.
    # Yao Dong, Xiaobo Guo, Ji Xiang, Kai Liu, Zhihao Tang: HyperspherE: An Embedding Method for Knowledge Graph
    #   Completion Based on Hypersphere. KSEM 2021: 517-528.
    # Xiaobo Guo, Neng Gao, Jun Yuan, Lin Zhao, Lei Wang, Sibo Cai: TransBidiFilter: Knowledge Embedding Based on a
    #   Bidirectional Filter. NLPCC (1) 2020: 232-243.


    # Regularizer: https://ojs.aaai.org/index.php/AAAI/article/view/20490
    # Another regularizer: Zhanqiu Zhang, Jianyu Cai, and Jie Wang. 2020a. Duality-induced regularizer for tensor
    #   factorization based knowledge graph completion. NIPS, 33.

    # New loss function (contrastive learning): Zhiping Luo, Wentao Xu, Weiqing Liu, Jiang Bian, Jian Yin, Tie-Yan Liu:
    #   KGE-CL: Contrastive Learning of Tensor Decomposition Based Knowledge Graph Embeddings. COLING 2022: 2598-2607.
    #   (The problem with this loss function is that it does not generalize to the rest of the papers and framework as
    #   it splits the triples into subparts.)
    # New loss function: Shoukang Han, Xiaobo Guo, Lei Wang, Zeyi Liu, Nan Mu: Adaptive-Skip-TransE Model: Breaking
    #   Relation Ambiguities for Knowledge Graph Embedding. KSEM (1) 2019: 549-560.








    # Neural networks
    # CapsE: https://arxiv.org/pdf/1808.04122v3.pdf
    # CapsE: https://aclanthology.org/N19-1226/
    # MDE has a MDENN version.
    # ConKB: https://aclanthology.org/N18-2053/
    # WGE: https://arxiv.org/abs/2112.09231
    # https://www.sciencedirect.com/science/article/abs/pii/S095070512200870X
    # https://www.sciencedirect.com/science/article/abs/pii/S0950705121004500
    # https://www.sciencedirect.com/science/article/pii/S0950705122012205
    # https://aclanthology.org/2020.emnlp-main.460/
    # https://arxiv.org/abs/2205.12102
    # R-GCN: Schlichtkrull, M., Kipf, T.N., Bloem, P., van den Berg, R., Titov, I., Welling, M.: Modeling relational
    #   data with graph convolutional networks. ESWC 2018. 593–607.
    # CACL: Oh, B., Seo, S., Lee, K.: Knowledge graph completion by context-aware convolutional learning with multi-hop
    #   neighborhoods. In: CIKM, pp. 257–266 (2018).
    # LogicENN: https://arxiv.org/pdf/1908.07141.pdf
    # CRNN: https://ieeexplore.ieee.org/document/8890615
    # NTransGH: https://www.sciencedirect.com/science/article/pii/S1877750318310172
    # SLM: R. Socher, D. Chen, C. D. Manning, and A. Y. Ng, ``Reasoning with neural tensor networks for knowledge base
    #   completion,'' NIPS, 2013.
    # NTN: Socher, Richard, Chen, Danqi, Manning, Christopher D., and Ng, Andrew Y. Reasoning with neural
    #           tensor networks for knowledge base completion. In NIPS, 2013.
    # ConvR: X. Jiang, Q. Wang, B. Wang, Adaptive convolution for multi-relational learning, in: Proc. NAACL-HLT, 2019.
    # RelNN: RelNN: A Deep Neural Model for Relational Learning. Seyed Mehran Kazemi, David Poole. AAAI, 2018.
    # Wang, S.; Wei, X.; dos Santos, C. N.; Wang, Z.; Nallapati, R.; Arnold, A.; Xiang, B.; Philip, S. Y.; and
    #   Cruz, I. F. 2021. Mixed-Curvature Multi-Relational Graph Neural Network for Knowledge Graph Completion. WWW.
    # Conv3D: Wenying Feng, Daren Zha, Lei Wang, Xiaobo Guo: Convolutional 3D Embedding for Knowledge Graph Completion.
    #   CSCWD 2022: 1197-1202
    # GSDM: Xiaobo Guo, Neng Gao, Nan Mu, Yao Dong, Lei Wang, Yuanye He: GSDM: A Gated Semantic Discriminating Model for
    #   Knowledge Graph Completion. CSCWD 2022: 1360-1365.
    # Xiaobo Guo, Fali Wang, Neng Gao, Zeyi Liu, Kai Liu: ConvMB: Improving Convolution-Based Knowledge Graph Embeddings
    #   by Adopting Multi-Branch 3D Convolution Filters. ISPA/BDCloud/SocialCom/SustainCom 2021: 382-389.
    # CoKE: Q. Wang, P. Huang, H. Wang, S. Dai, W. Jiang, J. Liu, Y. Lyu, Y. Zhu, H. Wu, CoKE:
    #   Contextualized knowledge graph embedding, 2019. --> https://github.com/PaddlePaddle/Research/blob/master/KG/CoKE/bin/model/coke.py
    # HittER: S. Chen, X. Liu, J. Gao, J. Jiao, R. Zhang, Y. Ji, HittER: Hierarchical transformers for knowledge graph
    #   embeddings, in: EMNLP, 2021, pp. 10395–10407. --> https://github.com/microsoft/HittER/blob/main/kge/model/trme.py#L190
    # CompGCN: Shikhar Vashishth, Soumya Sanyal, Vikram Nitin, and Partha Talukdar. 2020. Composition-based multi-
    #   relational graph convolutional networks. In ICLR.
    # Wenpeng Yin, Yadollah Yaghoobzadeh, and Hinrich Schütze. 2018. Recurrent one-hop predictions for reasoning over
    #   knowledge graphs. In COLING, 2369–2378.



    # ???
    # NoGE: https://dl.acm.org/doi/10.1145/3488560.3502183
    # https://arxiv.org/abs/1909.03821
    # https://www.hindawi.com/journals/sp/2020/7084958/
    # GAKE: https://aclanthology.org/C16-1062/ (even though it supports knowledge graphs, it is for regular graphs.)
    # TransG: https://aclanthology.org/P16-1219/ (talks about CRP and how to get M_r from it.)
    # Xiaobo Guo, Neng Gao, Lei Wang, Xin Wang: TransI: Translating Infinite Dimensional Embeddings Based on Trend
    #   Smooth Distance. KSEM (1) 2019: 511-523. (How do you compute the function?)
