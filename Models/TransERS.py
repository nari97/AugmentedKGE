import torch
from Models.TransE import TransE


class TransERS(TransE):

    def get_default_loss(self):
        return 'margin_limit'
