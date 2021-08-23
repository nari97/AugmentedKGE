import torch
import torch.nn as nn
import os

class BaseModule(nn.Module):

    def __init__(self):
        super(BaseModule, self).__init__()
        self.zero_const = nn.Parameter(torch.Tensor([0]))
        self.zero_const.requires_grad = False
        self.pi_const = nn.Parameter(torch.Tensor([3.14159265358979323846]))
        self.pi_const.requires_grad = False
        self.epoch = 0

    def load_checkpoint(self, path):
        dict = torch.load(os.path.join(path))
        if 'epoch' in dict.keys():
            self.epoch = dict.pop('epoch')
        self.load_state_dict(dict)
        self.eval()

    def save_checkpoint(self, path, epoch=0):
        dict = self.state_dict()
        dict['epoch'] = epoch
        torch.save(dict, path)