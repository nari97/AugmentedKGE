import torch
import torch.nn as nn
from .BaseModule import BaseModule


class Model(BaseModule):

    def __init__(self, ent_tot, rel_tot):
        super(Model, self).__init__()
        self.ent_tot = ent_tot
        self.rel_tot = rel_tot

    def normalize(self):
        raise NotImplementedError

    def forward(self):
        raise NotImplementedError

    def predict(self):
        raise NotImplementedError

    def get_batch(self, data, type):

        if type == "h":
            return data['batch_h']
        if type == "r":
            return data['batch_r']
        if type == "t":
            return data['batch_t']


    

    