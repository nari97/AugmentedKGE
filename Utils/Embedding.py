import torch
import torch.nn as nn


# Embedding redesign
class Embedding(nn.Module):
    """
    The embedding class contains all the functionalities required to create, initialize and normalize the embeddings.

    """

    def __init__(self, n_emb, n_dim, emb_type, name, init="xavier_uniform", init_params=[0, 1], seed=None):
        """Init function to create and initialize embeddings

        Args:
            n_emb (int): Number of embeddings
            n_dim (int or tuple): Dimension of embedding
            emb_type (str): Type of representation of embeddings: entity, relation or global.
            name (str): Name of the embedding to be accessed.
            init (str): Type of initialization for the embedding, 'xavier_uniform' uses the inbuilt xavier_uniform function and 'uniform' uses the inbuilt uniform function. Default: 'xavier_uniform'
            init_params (str): Contains the parameters required for initialization, for example, uniform initialization requires a lower and upper bound for each value. Default: None
            seed (int): seed to use for initialization
        """
        super(Embedding, self).__init__()

        self.n_emb = n_emb
        self.n_dim = n_dim
        self.emb_type = emb_type
        self.name = name

        self.init = init
        self.init_params = init_params
        self.emb = None
        self.create_embedding()
        self.init_embedding(seed)

    def create_embedding(self):
        """
        Creates an embedding based on the required size
        """
        if type(self.n_dim) is tuple:
            empty = torch.empty((self.n_emb, *self.n_dim), dtype=torch.float64)
        else:
            empty = torch.empty((self.n_emb, self.n_dim), dtype=torch.float64)
        self.emb = torch.nn.Parameter(empty)

    def init_embedding(self, seed):
        if seed is not None:
            torch.manual_seed(seed)
        """
        Initialises embeddings using the type of initialisation specified.
        """
        if self.init == "xavier_uniform":
            torch.nn.init.xavier_uniform_(self.emb.data)
        elif self.init == "xavier_normal":
            torch.nn.init.xavier_normal_(self.emb.data)
        elif self.init == "uniform":
            torch.nn.init.uniform_(self.emb.data, a=self.init_params[0], b=self.init_params[1])
        elif self.init == "kaiming_uniform":
            torch.nn.init.kaiming_uniform_(self.emb.data)

        # From time to time, using kaiming gives nan, we avoid that.
        if True in torch.isnan(self.emb.data):
            self.emb.data = torch.nan_to_num(self.emb.data)

    def get_embedding(self, data):
        """
        Returns the embeddings of the corresponding indices

        Args:
            batch (Tensor): Tensor containing the indices for which embeddings are required

        Returns:
            emb: Tensor of embeddings for the corresponding indices
        """
        return self.emb[data]
