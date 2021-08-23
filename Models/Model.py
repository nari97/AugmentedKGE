import torch
import torch.nn as nn
from Models.BaseModule import BaseModule


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

    def normalize(self):
        """
        Implement normalizations for parameters of the model
        """
        raise NotImplementedError

    def forward(self, data):
        """
        Calculate the scores for the given batch of triples

        Args:
            data (Tensor): Tensor containing the batch of triples for which scores are to be calculated

        Returns:
            score (Tensor): Tensor containing the scores for each triple
        """
        raise NotImplementedError

    def predict(self, data):
        """
        Calculate the scores for a given batch of triples during evaluation

        Args:
            data (Tensor): Tensor containing the batch of triples for which scores are to be calculated

        Returns:
            score (Tensor): Tensor containing the scores for each triple
        """
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


    

    