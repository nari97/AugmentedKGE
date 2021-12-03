import torch
import torch.nn as nn
from Utils.Embedding import Embedding
import os
import Utils.NormUtils as NormUtils

class Model(nn.Module):

    """
        Base class for all models to inherit
    """

    def __init__(self, ent_tot, rel_tot, dims, model_name, inner_norm = False):
        """
        Args:
            ent_tot (int): Total number of entites
            rel_tot (int): Total number of relations
        """
        super(Model, self).__init__()

        self.ent_tot = ent_tot
        self.rel_tot = rel_tot
        self.dims = dims
        self.norm_params = {"p" : 2, "dim" : -1, "maxnorm" : 1}
        self.inner_norm = inner_norm
        self.model_name = model_name
        self.epoch = 0
        self.embeddings = {"entity" : {}, "relation" : {}}
        self.ranks = None
        self.totals = None
        self.hyperparameters = None
        self.pi_const = torch.Tensor([3.14159265358979323846])
        

    def forward(self, data):
        """
        Calculate the scores for the given batch of triples

        Args:
            data (Tensor): Tensor containing the batch of triples for which scores are to be calculated

        Returns:
            score (Tensor): Tensor containing the scores for each triple
        """
    
        head_emb = self.get_head_embeddings(data)
        tail_emb = self.get_tail_embeddings(data)
        rel_emb = self.get_relation_embeddings(data)

        # if self.inner_norm:
        #     head_emb, tail_emb, rel_emb = self.inner_normalize(head_emb, tail_emb, rel_emb)

        score = self.returnScore(head_emb,rel_emb,tail_emb).flatten()

        return score

    def inner_normalize(self, head_emb, tail_emb, rel_emb):
        for key in head_emb:
            head_emb[key] = self.normalize(head_emb[key], "entity", key)

        for key in tail_emb:
            tail_emb[key] = self.normalize(tail_emb[key], "entity", key)

        for key in rel_emb:
            rel_emb[key] = self.normalize(rel_emb[key], "relation", key)

        return head_emb, tail_emb, rel_emb
                
    def predict(self, data):
        """
        Calculate the scores for a given batch of triples during evaluation

        Args:
            data (Tensor): Tensor containing the batch of triples for which scores are to be calculated

        Returns:
            score (Tensor): Tensor containing the scores for each triple
        """
        
        score = -self.forward(data)

        return score


    def returnScore(self, head_emb, rel_emb, tail_emb):
        raise NotImplementedError

    def get_batch(self, data, type):
        """
        Retrieves the head, relation or tail indices from the given data

        Args:
            data (Tensor): Tensor containing the batch of triples for which scores are to be calculated

        Returns:
            data: Tensor containing the indices of the head entities, relation or tail entities 
        """

        if type == "h":
            return data['batch_h']
        if type == "r":
            return data['batch_r']
        if type == "t":
            return data['batch_t']

    def create_embedding(self, dimension, emb_type, name, init = "xavier_uniform", init_params = [], normMethod = None, norm_params = []):
        total = 0

        if emb_type == "entity":
            total = self.ent_tot
        elif emb_type == "relation":
            total = self.rel_tot
        else:
            raise Exception("Type of embedding must be relation or entity")

        self.embeddings[emb_type][name] = Embedding(total, dimension, emb_type, name, init, init_params, normMethod, norm_params)

    

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

    def normalize(self, embedding = None, type = None, name = None):

        if self.inner_norm:
            norm_params = self.embeddings[type][name].norm_params
            normMethod = self.embeddings[type][name].normMethod

            if "p" in norm_params:
                p = norm_params["p"]
            else:
                p = 2

            if "dim" in norm_params:
                dim = norm_params["dim"]
            else:
                dim = -1

            if "maxnorm" in norm_params:
                maxnorm = norm_params["maxnorm"]
            else:
                maxnorm = 1


            if normMethod == "norm":
                embedding = NormUtils.normalize(embedding, p, dim)
            elif normMethod == "clamp":
                embedding = NormUtils.clamp_norm(embedding, p, dim, maxnorm=maxnorm)
            else:
                pass

            return embedding

        else:
            for key1 in self.embeddings:
                for key2 in self.embeddings[key1]:
                    self.embeddings[key1][key2].normalize()

    def load_checkpoint(self, path):
        dict = torch.load(os.path.join(path))
        if 'epoch' in dict.keys():
            self.epoch = dict.pop("epoch")
        self.embeddings = dict.pop("embeddings")
        self.ranks = dict.pop("ranks")
        self.totals = dict.pop("totals")
        self.hyperparameters = dict.pop("hyperparameters")

        self.eval()

    def save_checkpoint(self, path, epoch=0):
        dict = {"embeddings" : self.embeddings, "ranks" : self.ranks, "totals" : self.totals, "epoch" : epoch, "hyperparamters" : self.hyperparameters}
        
        torch.save(dict, path)

    def set_params(self, params):
        self.hyperparameters = params

    def get_params(self):
        return self.hyperparameters

    def get_name(self):
        s = self.model_name
        for p in self.hyperparameters:
            s = s + "_" + p + "_" + str(self.hyperparameters[p])
        return s

    def register_params(self):

        for key1 in self.embeddings:
            for key2 in self.embeddings[key1]:
                self.register_parameter(key1+"_"+key2, self.embeddings[key1][key2].emb.weight)