import torch
from Models.TransE import TransE


class TransEFT(TransE):
    """
    Jun Feng, Minlie Huang, Mingdong Wang, Mantong Zhou, Yu Hao, Xiaoyan Zhu: Knowledge Graph Embedding by Flexible
        Translation. KR 2016: 557-560.
    # TODO Can we implement all the models in the same manner?
    """
    def _calc(self, h, r, t):
        return -torch.linalg.norm((h + r) * t + h * (t - r), dim=-1, ord=self.pnorm)
