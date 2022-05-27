import torch
import torch.nn as nn
from Utils.Embedding import Embedding
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
        self.ranks = None
        self.totals = None
        self.hyperparameters = None
        self.custom_constraints = []
        self.scale_constraints = []
        self.current_batch = None
        self.current_global_embeddings = {}

    def set_use_gpu(self, use_gpu):
        self.use_gpu = use_gpu

    def get_model_name(self):
        return type(self).__name__.lower()

    def initialize_model(self):
        raise NotImplementedError

    def get_default_loss(self):
        raise NotImplementedError

    def forward(self, data):
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

        self.embeddings_to_gpu()
        scores = self.return_score().flatten()
        self.embeddings_to_cpu()

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

            for name in self.current_global_embeddings.keys():
                self.current_global_embeddings[name] = self.move_to_cpu(self.current_global_embeddings[name])
            self.current_global_embeddings = {}

            torch.cuda.empty_cache()

    def apply_normalization(self):
        for emb_type in self.embeddings:
            for name in self.embeddings[emb_type]:
                emb = self.embeddings[emb_type][name]
                norm_info = self.embeddings_normalization[emb_type][name]

                if norm_info['method'] is 'norm':
                    p = norm_info['params']['p']
                    dim = norm_info['params']['dim']
                    # This is in place.
                    emb.emb.data = torch.nn.functional.normalize(emb.emb.data, p, dim)
                elif norm_info['method'] is None:
                    pass
                else:
                    print('Warning! You specified a norm method not recognized: ', norm_info['method'],
                          '; are you sure about this?')
                    pass

    def register_scale_constraint(self, emb_type, name, p, z=1):
        self.scale_constraints.append({'emb_type': emb_type, 'name': name, 'p': p, 'z': z})

    # Gets the embedding and applies the constraint ||x||_y<=z
    def scale_constraint(self, emb, p, z=1):
        return nn.functional.normalize(emb, p) - z

    def register_custom_constraint(self, c):
        self.custom_constraints.append(c)

    # Apply the constraints as regularization (to be added to the loss) using either L1 or L2.
    def regularization(self, data, reg_type='L2'):
        head_emb = self.get_head_embeddings(data)
        tail_emb = self.get_tail_embeddings(data)
        rel_emb = self.get_relation_embeddings(data)

        all_constraints = {'custom': self.custom_constraints, 'scale': self.scale_constraints}

        reg, total = 0, 0
        for key in all_constraints.keys():
            for c in all_constraints[key]:
                v = []
                if key is 'custom':
                    v.append(c(head_emb, rel_emb, tail_emb))
                elif key is 'scale':
                    if c['emb_type'] == 'entity':
                        v.append(self.scale_constraint(head_emb[c['name']], c['p'], c['z']))
                        v.append(self.scale_constraint(tail_emb[c['name']], c['p'], c['z']))
                    elif c['emb_type'] == 'relation':
                        v.append(self.scale_constraint(rel_emb[c['name']], c['p'], c['z']))

                for x in v:
                    if reg_type is 'L1':
                        x = torch.abs(x)
                    elif reg_type is 'L2':
                        x = torch.pow(x, 2)
                    reg += torch.sum(x)
                    total += 1

        # See https://towardsdatascience.com/understanding-the-scaling-of-l%C2%B2-regularization-in-the-context-of-neural-networks-e3d25f8b50db
        if reg_type is 'L2':
            reg = .5*reg
        if total > 0:
            reg /= total

        return reg

    def predict(self, data):
        """
        Calculate the scores for a given batch of triples during evaluation

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
        self.embeddings_to_gpu()
        scores = -self.return_score(is_predict=True).flatten()
        self.embeddings_to_cpu()

        return scores

    def return_score(self):
        raise NotImplementedError

    def create_embedding(self, dimension, emb_type=None, name=None, register=True, init="xavier_uniform",
                         init_params=[], norm_method=None, norm_params={"p": 2, "dim": -1}):
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
        else:
            return emb

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
