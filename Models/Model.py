import torch
import torch.nn as nn
from Models.BaseModule import BaseModule
from Utils.Embedding import Embedding


class Model(BaseModule):

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
        self.embeddings = {"entity" : {}, "relation" : {}}

        self.ranks = None
        self.totals = None
        

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

        score = self.returnScore(head_emb,rel_emb,tail_emb).flatten()

        return score

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

    def create_embedding(self, total, dimension, emb_type, name, init = "xavier_uniform", init_params = [], normMethod = "none", norm_params = []):

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

    def normalize(self):

        for key1 in self.embeddings:
            for key2 in self.embeddings[key1]:
                self.embeddings[key1][key2].normalize()

    