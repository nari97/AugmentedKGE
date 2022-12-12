import torch
import torch.nn as nn
import time
from ..Utils.Embedding import Embedding
import os


class Model(nn.Module):
    """
        Base class for all models to inherit
    """

    def __init__(self, ent_tot, rel_tot):
        """
        Args:
            ent_tot (int): Total number of entites
            rel_tot (int): Total number of relations
        """
        super(Model, self).__init__()

        self.ent_tot = ent_tot
        self.rel_tot = rel_tot
        self.use_gpu = False

        self.epoch = 0

        self.embeddings = {"entity": {}, "relation": {}, "global" : {}}
        self.embeddings_normalization = {"entity": {}, "relation": {}, "global" : {}}
        self.embeddings_regularization = {"entity": {}, "relation": {}, "global": {}}
        # This is for complex numbers.
        self.embeddings_regularization_complex = {"entity": [], "relation": [], "global": []}
        # This regularization is executed once for the current batch.
        self.onthefly_regularization = []

        self.ranks = None
        self.totals = None
        self.hyperparameters = None

        self.custom_constraints = []
        self.scale_constraints = []
        # These constraints are executed once for the current batch.
        self.onthefly_constraints = []

        self.current_batch = None
        self.current_data = None
        self.current_global_embeddings = {}

    def set_use_gpu(self, use_gpu):
        self.use_gpu = use_gpu
        for key1 in self.embeddings:
            for key2 in self.embeddings[key1]:
                self.embeddings[key1][key2] = self.embeddings[key1][key2].to("cuda")

    def get_model_name(self):
        return type(self).__name__.lower()

    def initialize_model(self):
        raise NotImplementedError

    def get_default_loss(self):
        raise NotImplementedError

    def forward(self, data):
        return self.get_scores(data)

    def get_scores(self, data, is_predict=False):
        """
                Calculate the scores for the given batch of triples

                Args:
                    data (Tensor): Tensor containing the batch of triples for which scores are to be calculated

                Returns:
                    score (Tensor): Tensor containing the scores for each triple
                """
        head_emb = self.get_head_embeddings(data)
        rel_emb = self.get_relation_embeddings(data)
        tail_emb = self.get_tail_embeddings(data)
        self.get_global_embeddings()

        self.current_batch = (head_emb, rel_emb, tail_emb)
        self.current_data = data

        #self.embeddings_to_gpu()
        scores = self.return_score(is_predict=is_predict).flatten()

        #self.embeddings_to_cpu()

        return scores

    def embeddings_to_gpu(self):
        # Move to GPU?
        if self.use_gpu:
            (head_emb, rel_emb, tail_emb) = self.current_batch
            for emb in [head_emb, tail_emb, rel_emb]:
                for name in emb.keys():
                    emb[name] = self.move_to_gpu(emb[name])
            self.current_batch = (head_emb, rel_emb, tail_emb)

            for name in self.current_global_embeddings.keys():
                self.current_global_embeddings[name] = self.move_to_gpu(self.current_global_embeddings[name])

    def embeddings_to_cpu(self):
        if self.use_gpu:
            (head_emb, rel_emb, tail_emb) = self.current_batch
            for emb in [head_emb, tail_emb, rel_emb]:
                for name in emb.keys():
                    emb[name] = self.move_to_cpu(emb[name])
            self.current_batch = None
            self.current_data = None

            for name in self.current_global_embeddings.keys():
                self.current_global_embeddings[name] = self.move_to_cpu(self.current_global_embeddings[name])
            self.current_global_embeddings = {}

            torch.cuda.empty_cache()

    # This is called before starting the batch and gradients.
    def start_batch(self):
        self.apply_normalization()

    # This is called when the batch is done.
    def end_batch(self):
        pass

    def apply_normalization(self):
        for emb_type in self.embeddings:
            for name in self.embeddings[emb_type]:
                emb = self.embeddings[emb_type][name]
                norm_info = self.embeddings_normalization[emb_type][name]

                if norm_info['method'] == 'norm':
                    p = norm_info['params']['p']
                    dim = norm_info['params']['dim']
                    # This is in place.
                    emb.emb.data = torch.nn.functional.normalize(emb.emb.data, p, dim)
                elif norm_info['method'] == 'rescaling':
                    a = norm_info['params']['a']
                    b = norm_info['params']['b']

                    # This is in place.
                    min, max = emb.emb.data.min(1, keepdim=True)[0], emb.emb.data.max(1, keepdim=True)[0]
                    emb.emb.data -= min
                    emb.emb.data /= (max - min)

                    emb.emb.data *= b - a
                    emb.emb.data += a
                elif norm_info['method'] is None:
                    pass
                else:
                    print('Warning! You specified a norm method not recognized: ', norm_info['method'],
                          '; are you sure about this?')
                    pass

    def register_scale_constraint(self, emb_type, name, p=2, z=1, ctype='le'):
        self.scale_constraints.append({'emb_type': emb_type, 'name': name, 'p': p, 'z': z, 'ctype': ctype})

    # Applies the constraint ||x||_y<=z when ctype='le', and ||x||_y>=z when ctype='ge'.
    # Check, for instance, TransH and GTrans on how these are applied.
    def scale_constraint(self, emb, p=2, z=1, ctype='le'):
        if ctype == 'le':
            constraint = torch.pow(torch.linalg.norm(emb, dim=-1, ord=p), 2) - z
        elif ctype == 'ge':
            constraint = z - torch.pow(torch.linalg.norm(emb, dim=-1, ord=p), 2)
        return torch.maximum(constraint, torch.zeros_like(constraint))

    def register_custom_constraint(self, c):
        self.custom_constraints.append(c)

    # Apply the constraints to be added to the loss.
    def constraints(self, data):
        head_emb = self.get_head_embeddings(data)
        tail_emb = self.get_tail_embeddings(data)
        rel_emb = self.get_relation_embeddings(data)

        all_constraints = {'custom': self.custom_constraints, 'scale': self.scale_constraints,
                           'onthefly': self.onthefly_constraints}

        constraints = 0
        for key in all_constraints.keys():
            for c in all_constraints[key]:
                v = []
                if key is 'custom':
                    v.append(c(head_emb, rel_emb, tail_emb))
                elif key is 'scale':
                    if c['emb_type'] == 'entity':
                        v.append(self.scale_constraint(head_emb[c['name']], c['p'], c['z'], c['ctype']))
                        v.append(self.scale_constraint(tail_emb[c['name']], c['p'], c['z'], c['ctype']))
                    elif c['emb_type'] == 'relation':
                        v.append(self.scale_constraint(rel_emb[c['name']], c['p'], c['z'], c['ctype']))
                elif key is 'onthefly':
                    v.append(c)

                for x in v:
                    constraints += torch.sum(x)

        # Clear on-the-fly constraints.
        self.onthefly_constraints = []

        return constraints

    # Apply regularization.
    def regularization(self, data, reg_type='L2'):
        reg, total = 0, 0

        # TODO This can be improved by taking only the embeddings mentioned in the batch!
        for etype in self.embeddings_regularization.keys():
            for ename in self.embeddings_regularization[etype].keys():
                reg_params = self.embeddings_regularization[etype][ename]
                r = self.apply_individual_regularization(self.embeddings[etype][ename].emb.data, reg_type, reg_params)
                reg += torch.sum(r)
                total += len(r)

        for etype in self.embeddings_regularization_complex.keys():
            for d in self.embeddings_regularization_complex[etype]:
                real_part, img_part = self.embeddings[etype][d["real_part"]].emb.data, \
                                      self.embeddings[etype][d["img_part"]].emb.data
                reg_params = d["params"]
                complex = torch.view_as_complex(torch.stack((real_part, img_part), dim=-1))
                r = self.apply_individual_regularization(complex, reg_type, reg_params)
                reg += torch.sum(r)
                total += len(r)

        for (x, reg_params) in self.onthefly_regularization:
            r = self.apply_individual_regularization(x, reg_type, reg_params)
            reg += torch.sum(r)
            total += len(r)

        # Clear on-the-fly regularization.
        self.onthefly_regularization = []

        if reg_type is 'L2':
            reg = 1/2 * reg
        elif reg_type is 'L3':
            reg = 1/3 * reg

        if total > 0:
            reg /= total

        return reg

    def apply_individual_regularization(self, x, reg_type, reg_params):
        if reg_type is 'L1':
            p = 1
            f = torch.abs
        elif reg_type is 'L2':
            p = 2
            f = torch.pow
        elif reg_type is 'L3':
            p = 3
            f = torch.pow

        # It can be fro, for instance.
        if "p" in reg_params.keys():
            prev_p = p
            p = reg_params["p"]

        if "transform" in reg_params.keys():
            x = reg_params["transform"](x)
        r = reg_params["norm"](x, dim=reg_params["dim"], ord=p)

        if "p" in reg_params.keys():
            p = prev_p

        if p == 1:
            r = f(r)
        else:
            r = f(r, p)

        return r

    def predict(self, data):
        return self.get_scores(data, is_predict=True)

    def return_score(self):
        raise NotImplementedError

    def register_onthefly_regularization(self, x, reg_params={"norm": torch.linalg.norm, "dim": -1}):
        self.onthefly_regularization.append((x, reg_params))

    def create_embedding(self, dimension, emb_type=None, name=None, register=True,
                         init="xavier_uniform", init_params=[],
                         norm_method=None, norm_params={"p": 2, "dim": -1},
                         reg=False, reg_params={"norm": torch.linalg.norm, "dim": -1}):
        if emb_type == "entity":
            total = self.ent_tot
        elif emb_type == "relation":
            total = self.rel_tot
        elif emb_type == "global":
            total = 1
        else:
            raise Exception("Type of embedding must be relation or entity")

        emb = Embedding(total, dimension, emb_type, name, init, init_params)
        if register:
            self.embeddings[emb_type][name] = emb
            self.embeddings_normalization[emb_type][name] = {'method': norm_method, 'params': norm_params}
            self.register_parameter(emb_type + '_' + name, self.embeddings[emb_type][name].emb)

        if reg:
            self.embeddings_regularization[emb_type][name] = reg_params

        return emb

    def register_complex_regularization(self, emb_type=None, name_real=None, name_img=None,
                                         reg_params={"norm": torch.linalg.norm, "dim": -1}):
        self.embeddings_regularization_complex[emb_type].append({"real_part": name_real, "img_part": name_img,
                                                                 "params": reg_params})

    def move_to_gpu(self, emb):
        return emb.to(torch.device('cuda'))

    def move_to_cpu(self, emb):
        return emb.to(torch.device('cpu'))

    def get_embedding(self, emb_type, name):
        return self.embeddings[emb_type][name]

    def get_global_embeddings(self):
        self.current_global_embeddings = {}
        for emb in self.embeddings["global"]:
            self.current_global_embeddings[emb] = self.embeddings["global"][emb].emb
        return self.current_global_embeddings

    def get_head_embeddings(self, data):
        head_embeddings = {}
        for emb in self.embeddings["entity"]:
            head_embeddings[emb] = self.embeddings["entity"][emb].get_embedding(data["batch_h"])
        return head_embeddings

    def get_tail_embeddings(self, data):
        tail_embeddings = {}
        for emb in self.embeddings["entity"]:
            tail_embeddings[emb] = self.embeddings["entity"][emb].get_embedding(data["batch_t"])
        return tail_embeddings

    def get_relation_embeddings(self, data):
        relation_embeddings = {}
        for emb in self.embeddings["relation"]:
            relation_embeddings[emb] = self.embeddings["relation"][emb].get_embedding(data["batch_r"])
        return relation_embeddings

    def load_checkpoint(self, path):
        dic = torch.load(os.path.join(path))
        if 'epoch' in dic.keys():
            self.epoch = dic.pop("epoch")
        self.embeddings = dic.pop("embeddings")
        self.ranks = dic.pop("ranks")
        self.totals = dic.pop("totals")
        self.hyperparameters = dic.pop("hyperparameters")

        self.eval()

    def save_checkpoint(self, path, epoch=0):
        to_save = {"embeddings": self.embeddings, "ranks": self.ranks, "totals": self.totals,
                   "epoch": epoch, "hyperparameters": self.hyperparameters}

        torch.save(to_save, path)

    def set_params(self, params):
        self.hyperparameters = params

    def get_params(self):
        return self.hyperparameters
