import torch
import torch.nn as nn
from Utils.utils import clamp_norm

class Embedding(nn.Module):
    """
    The embedding class contains all the functionalities required to create, initialize and normalize the embeddings.

    """

    def __init__(self, n_emb, n_dim , rep = 'real', c_rep = 'complex', init = 'xavier_uniform', init_params = None):
        """Init function to create and initialize embeddings

        Args:
            n_emb (int): Number of embeddings
            n_dim (int): Dimension of embedding
            rep (str): Type of representation of embeddings, 'real' for embeddings containing real numbers and 'complex' for embeddings containing complex numbers. Default : 'real'
            c_rep (str): For complex representations of embeddings, 'real' represents each individual dimension in the embedding as a vector [x, y] where the complex number c = x+iy, 'complex' represents each individual dimension as a complex number x+iy. Default: 'complex'
            init (str): Type of initialization for the embedding, 'xavier_uniform' uses the inbuilt xavier_uniform function and 'uniform' uses the inbuilt uniform function. Default: 'xavier_uniform'
            init_params (str): Contains the parameters required for initialization, for example, uniform initialization requires a lower and upper bound for each value. Default: None
        """
        
        super(Embedding, self).__init__()

        self.n_emb = n_emb
        self.n_dim = n_dim
        self.rep = rep
        self.c_rep = c_rep

        self.emb = None
        self.init = init

        self.init_params = init_params
        
        self.create_embedding()
        self.init_embedding()


    def create_embedding(self):
        """
        Creates an embedding based on the required size

        """

        if self.rep == 'real':
            self.emb = torch.nn.Embedding(self.n_emb, self.n_dim)

        if self.rep == 'complex':

            if self.c_rep == 'real':
                self.emb = torch.nn.Embedding(self.n_emb, self.n_dim*2)
            else:
                self.emb = torch.nn.Embedding(self.n_emb, self.n_dim)

    def init_embedding(self):
        """
        Initialises embeddings using the type of initialisation specified. Currently supports xavier_uniform and uniform initialisations

        """

        if self.init == 'xavier_uniform':
            self.emb.weight.data = torch.nn.init.xavier_uniform_(self.emb.weight.data)
        if self.init == 'uniform':
            self.emb.weight.data = torch.nn.init.uniform_(self.emb.weight.data, a = self.init_params[0], b = self.init_params[1])

    def normalize(self, norm = 'norm', p = 2, dim = -1, maxnorm = 1):

        """
        Applies L1 or L2 normalisation on the embedding belonging to the class

        Args:
            norm (str): Two types of normalisation, for constraints where x is the embedding and ||x|| = 1, use norm = 'norm', and ||x||<=1 use norm = 'clamp'. Default: 'norm'
            p (int): The exponent value in normalisation. Default: 2
            dim (int): The dimension to normalize. Default: -1

        """
        if norm == 'norm':
            self.emb.weight.data = torch.nn.functional.normalize(self.emb.weight.data, p, dim)

        else:
            self.emb.weight.data = clamp_norm(self.emb.weight.data, p = 2, dim = -1, maxnorm=maxnorm)

    def get_embedding(self, batch):

        """
        Returns the embeddings of the corresponding indices

        Args:
            batch (Tensor): Tensor containing the indices for which embeddings are required

        Returns:
            emb: Tensor of embeddings for the corresponding indices
        """

        return self.emb(batch)