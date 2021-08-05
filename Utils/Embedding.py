import torch

class Embedding:

    def __init__(self, n_emb, n_dim , rep = 'real', c_rep = 'complex', init = 'xavier_uniform'):

        self.n_emb = n_emb
        self.n_dim = n_dim
        self.rep = rep
        self.c_rep = c_rep

        self.emb = None
        self.init = init
        
        self.create_embedding()
        self.init_embedding()


    def create_embedding(self):

        if self.rep == 'real':
            self.emb = torch.nn.Embedding(self.n_emb, self.n_dim)

        if self.rep == 'complex':

            if self.c_rep == 'real':
                self.emb = torch.nn.Embedding(self.n_emb, self.n_dim*2)
            else:
                self.emb = torch.nn.Embedding(self.n_emb, self.n_dim)

    def init_embedding(self):

        if self.init == 'xavier_uniform':
            self.emb.weight.data = torch.nn.init.xavier_uniform_(self.emb.weight.data)

    def normalize(self, norm = 'norm', p = 2, dim = -1, maxnorm = 1):

        self.emb.weight.data = torch.nn.functional.normalize(self.emb.weight.data, p, dim)

        if norm == 'clamp':
            self.emb.weight.data = torch.clamp(self.emb.weight.data, max = maxnorm)

    def get_embedding(self, batch):

        return self.emb(batch)