import torch
from Models.Model import Model


class TransMVG(Model):
    """
    Xiaobo Guo, Neng Gao, Jun Yuan, Xin Wang, Lei Wang, Di Kang: TransMVG: Knowledge Graph Embedding Based on
        Multiple-Valued Gates. WISE (1) 2020: 286-301.
    """
    def __init__(self, ent_total, rel_total, dim, norm=2):
        """
            dim (int): Number of dimensions for embeddings
            norm (int): L1 or L2 norm. Default: 2.
        """
        super(TransMVG, self).__init__(ent_total, rel_total)
        self.dim = dim
        self.pnorm = norm

        # Fully connected layers
        self.fch = torch.nn.Linear(dim*2, dim, dtype=torch.float64)
        self.fct = torch.nn.Linear(dim*2, dim, dtype=torch.float64)

    def get_default_loss(self):
        # Eq. (17).
        return 'margin'

    def get_score_sign(self):
        # It is a distance (norm).
        return -1

    def initialize_model(self):
        self.create_embedding(self.dim, emb_type="entity", name="e")
        self.create_embedding(self.dim, emb_type="entity", name="bh")
        self.create_embedding(self.dim, emb_type="entity", name="bt")
        # "The elements in Uh and Ut are independent with each other and are sampled from a uniform distribution (0,1)
        #   respectively."
        self.create_embedding(self.dim, emb_type="entity", name="uh", init_method="uniform", init_params=[0, 1])
        self.create_embedding(self.dim, emb_type="entity", name="ut", init_method="uniform", init_params=[0, 1])
        self.create_embedding(self.dim, emb_type="relation", name="r")
        # In the paper is proposed as a hyperparameter. We include it as parameter.
        self.create_embedding(1, emb_type="global", name="tau")

    def gate(self, e, u, r, b, layer, tau):
        # Eqs. (10), (11), (12), (13), (14) and (15). Stack inputs.
        u_clamped = torch.clamp(u, min=1e-10, max=1-1e-10)
        noise = torch.log(u_clamped) - torch.log(1 - u_clamped)
        return e * torch.sigmoid((layer(torch.cat([e, r], 1)) + b + noise)/tau)

    def _calc(self, h, uh, bh, r, t, ut, bt, tau):
        # These are h_r and t_r in Eqs. (14) and (15).
        hgate = self.gate(h, uh, r, bh, self.fch, tau)
        tgate = self.gate(t, ut, r, bt, self.fct, tau)

        return torch.linalg.norm(hgate + r - tgate, dim=-1, ord=self.pnorm)

    def return_score(self, is_predict=False):
        (head_emb, rel_emb, tail_emb) = self.current_batch

        h, uh, bh = head_emb["e"], head_emb["uh"], head_emb["bh"]
        t, ut, bt = tail_emb["e"], tail_emb["ut"], tail_emb["bt"]
        r = rel_emb["r"]
        tau = self.current_global_embeddings['tau']

        return self._calc(h, uh, bh, r, t, ut, bt, tau)
