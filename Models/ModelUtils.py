from Models.TransE import TransE
from Models.TransH import TransH
from Models.TransD import TransD
from Models.SimplE import SimplE
from Models.HolE import HolE
from Models.RotatE import RotatE
from Models.DistMult import DistMult
from Models.ComplEx import ComplEx


from Strategy.NegativeSampling import NegativeSampling
from Loss.MarginLoss import MarginLoss
from Loss.SigmoidLoss import SigmoidLoss
from Loss.SoftplusLoss import SoftplusLoss
from Loss.MarginSigmoidLoss import MarginSigmoidLoss
from Loss.NegativeSamplingLoss import NegativeSamplingLoss

import os
import ast

class ModelUtils:
    def __init__(self, model_name, params):
        self.model_name = model_name
        self.params = params

    # https://stackoverflow.com/questions/2859674/converting-python-list-of-strings-to-their-type
    @staticmethod
    def tryeval(val):
        try:
            val = ast.literal_eval(val)
        except ValueError:
            pass
        return val

    @staticmethod
    def get_params(model_file):
        filename, ext = os.path.splitext(model_file)
        # No extension!
        s = filename.split("_")

        paramaters = {}
        for i in range(1, len(s), 2):
            p = s[i]
            value = s[i + 1]

            if p=='trial':
                break
            paramaters[p] = ModelUtils.tryeval(value)
        return paramaters

    def get_name(self):
        s = self.model_name
        for p in self.params:
            s = s + "_" + p + "_" + str(self.params[p])
        return s

    def get_model(self, ent_total, rel_total, batch_size, inner_norm = False):
        m = None
        if self.model_name == "transe":
            m = TransE(
                ent_total=ent_total,
                rel_total=rel_total,
                dims=self.params["dim"],
                norm=self.params["pnorm"],
                inner_norm = inner_norm)
        elif self.model_name == "transh":
            m = TransH(
                ent_total=ent_total,
                rel_total=rel_total,
                dims=self.params["dim"],
                norm=self.params["pnorm"],
                inner_norm = inner_norm)
        elif self.model_name == "transd":
            m = TransD(
                ent_total=ent_total,
                rel_total=rel_total,
                dim_e=self.params["dime"],
                dim_r=self.params["dimr"],
                norm=self.params["pnorm"],
                inner_norm = inner_norm)
        elif self.model_name == "transr":
            m = TransR(
                ent_total=ent_total,
                rel_total=rel_total,
                dim_e=self.params["dime"],
                dim_r=self.params["dimr"],
                p_norm=self.params["pnorm"],
                norm_flag=self.params["norm"],
                inner_norm = inner_norm)
        elif self.model_name == "rescal":
            m = RESCAL(
                ent_tot=ent_total,
                rel_tot=rel_total,
                dim=self.params["dim"],
                inner_norm = inner_norm)
        elif self.model_name == "distmult":
            m = DistMult(
                ent_total=ent_total,
                rel_total=rel_total,
                dims=self.params["dim"],
                inner_norm = inner_norm)
        elif self.model_name == "complex":
            m = ComplEx(
                ent_total=ent_total,
                rel_total=rel_total,
                dims=self.params["dim"],
                inner_norm = inner_norm)
        elif self.model_name == "hole":
            m = HolE(
                ent_total=ent_total,
                rel_total=rel_total,
                dims=self.params["dim"],
                inner_norm = inner_norm)
        elif self.model_name == "simple":
            m = SimplE(
                ent_total=ent_total,
                rel_total=rel_total,
                dims=self.params["dim"],
                inner_norm = inner_norm)
        elif self.model_name == "analogy":
            m = Analogy(
                ent_total=ent_total,
                rel_total=rel_total,
                dims=self.params["dim"],
                inner_norm = inner_norm)
        elif self.model_name == "rotate":
            m = RotatE(
                ent_total=ent_total,
                rel_total=rel_total,
                dims=self.params["dim"],
                inner_norm = inner_norm)
        elif self.model_name == "amie":
            m = Amie()

        if self.model_name == "transe" or self.model_name == "transh" or self.model_name == "transd" or \
                self.model_name == "rescal" or self.model_name == "transr":
            loss=MarginLoss(margin=self.params["gamma"])
            print ('Loss : Margin Loss')
        elif self.model_name == 'hole' or self.model_name == 'distmult':
            loss = MarginSigmoidLoss(margin = self.params["gamma"])
            print ('Loss : Margin Sigmoid Loss')
        elif self.model_name == "rotate":
            loss = NegativeSamplingLoss(margin = self.params["gamma"])
            print ('Loss : Negative Sampling Loss')
        elif self.model_name == "analogy":
            print ('Loss : Sigmoid Loss')
            loss = SigmoidLoss()
        else:
            print ('Loss : Softplus Loss')
            loss = SoftplusLoss()
        
        for name, p in m.named_parameters():
            print (name)
        
        return NegativeSampling(
                    model=m,
                    loss=loss,
                    batch_size=batch_size)