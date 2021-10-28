import torch
import torch.nn as nn
from Utils.utils import clamp_norm

#Embedding redesign
class Embedding(nn.Module):
    """
    The embedding class contains all the functionalities required to create, initialize and normalize the embeddings.

    """

    def __init__(self, n_emb, n_dim, emb_type, name, init = "xavier_uniform", init_params = [0,1], normMethod = "norm", norm_params = []):
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
        self.emb_type = emb_type
        self.name = name

        self.init = init
        self.init_params = init_params
        self.emb = None

        self.normMethod = normMethod
        self.norm_params = norm_params
        
        
        self.create_embedding()
        self.init_embedding()


    def create_embedding(self):
        """
        Creates an embedding based on the required size

        """
        
        self.emb = torch.nn.Embedding(self.n_emb, self.n_dim)

    def init_embedding(self):
        """
        Initialises embeddings using the type of initialisation specified. Currently supports xavier_uniform and uniform initialisations

        """

        if self.init == "xavier_uniform":
            self.emb.weight.data = torch.nn.init.xavier_uniform_(self.emb.weight.data)
        if self.init == "uniform":
            self.emb.weight.data = torch.nn.init.uniform_(self.emb.weight.data, a = self.init_params[0], b = self.init_params[1])

    def normalize(self):

        """
        Applies L1 or L2 normalisation on the embedding belonging to the class

        Args:
            norm (str): Two types of normalisation, for constraints where x is the embedding and ||x|| = 1, use norm = 'norm', and ||x||<=1 use norm = 'clamp'. Default: 'norm'
            p (int): The exponent value in normalisation. Default: 2
            dim (int): The dimension to normalize. Default: -1

        """

        if "p" in self.norm_params:
            p = self.norm_params["p"]
        else:
            p = 2

        if "dim" in self.norm_params:
            dim = self.norm_params["dim"]
        else:
            dim = -1

        if "maxnorm" in self.norm_params:
            maxnorm = self.norm_params["maxnorm"]
        else:
            maxnorm = 1


        if self.normMethod == "norm":
            self.emb.weight.data = torch.nn.functional.normalize(self.emb.weight.data, p, dim)
        elif self.normMethod == "clamp":
            self.emb.weight.data = clamp_norm(self.emb.weight.data, p = 2, dim = -1, maxnorm=maxnorm)
        else:
            pass

    def get_embedding(self, data):

        """
        Returns the embeddings of the corresponding indices

        Args:
            batch (Tensor): Tensor containing the indices for which embeddings are required

        Returns:
            emb: Tensor of embeddings for the corresponding indices
        """
        return self.emb(data)