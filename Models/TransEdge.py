import torch
from Models.Model import Model


class TransEdge(Model):
    """
    Zequn Sun, Jiacheng Huang, Wei Hu, Muhao Chen, Lingbing Guo, Yuzhong Qu: TransEdge: Translating Relation-
        Contextualized Embeddings for Knowledge Graphs. ISWC (1) 2019: 612-629.
    """
    def __init__(self, ent_total, rel_total, dim, norm=1, variant='cc'):
        """
            dim (int): Number of dimensions for embeddings
            norm (int): L1 or L2 norm. Default: 1. "We adopt L2-norm in the energy function."
            variant can be either cc (context compression) or cp (context projection).
        """
        super(TransEdge, self).__init__(ent_total, rel_total)
        self.dim = dim
        self.pnorm = norm
        self.variant = variant

        if self.variant == 'cc':
            # The original paper does not talk about the hidden dimension. We use the same as ConvE.
            hidden_dim = 32

            # We have three multilayer perceptrons (MLPs). We declare dropout and norms for each layer.
            self.mlp2 = TransEdge.register_mlp(self.dim*2, hidden_dim)
            self.mlp3 = TransEdge.register_mlp(self.dim*2, hidden_dim)
            self.mlp1 = TransEdge.register_mlp(hidden_dim, self.dim)
        elif self.variant == 'cp':
            self.mlp = TransEdge.register_mlp(self.dim*2, self.dim)

    # Input dimension, out dimension, batch normalization (either 1d or 2d), dropout rate.
    @staticmethod
    def register_mlp(input, output, norm=1, dropout_rate=.2):
        fc = torch.nn.Linear(input, output, dtype=torch.float64)
        # The paper does not propose batch normalization or dropout rate, but we will use them.
        #if norm == 1:
        # TODO I think all of them are 1d.
        norm = torch.nn.BatchNorm1d(output, dtype=torch.float64)
        #elif norm == 2:
        #    norm = torch.nn.BatchNorm2d(output, dtype=torch.float64)
        dropout = torch.nn.Dropout(dropout_rate)
        # This is the activation proposed by the paper.
        activation = torch.tanh
        return (fc, norm, dropout, activation)

    @staticmethod
    def apply_mlp(mlp, input):
        (fc, norm, dropout, activation) = mlp
        return activation(norm(dropout(fc(input))))

    def get_default_loss(self):
        # Eq. (5).
        return 'limit'

    def get_score_sign(self):
        # It is a distance (norm).
        return -1

    def initialize_model(self):
        self.create_embedding(self.dim, emb_type="entity", name="e")
        self.create_embedding(self.dim, emb_type="entity", name="ec")
        self.create_embedding(self.dim, emb_type="relation", name="r")

    def _calc(self, h, hc, r, t, tc, is_predict):
        if self.variant == 'cc':
            # Eq. (3).
            psi = TransEdge.apply_mlp(self.mlp1, TransEdge.apply_mlp(self.mlp2, torch.concat([hc, r], 1)) +
                                      TransEdge.apply_mlp(self.mlp3, torch.concat([tc, r], 1)))
        elif self.variant == 'cp':
            wht = TransEdge.apply_mlp(self.mlp, torch.concat([hc, tc], 1))

            if not is_predict:
                # The paper says ||wht||=1; norm not specified (assuming 2); we implement ||wht||_2<=1 and ||wht||_2>=1.
                self.onthefly_constraints.append(self.scale_constraint(wht))
                self.onthefly_constraints.append(self.scale_constraint(wht, ctype='ge'))

            # Eq. (4).
            # wht^T*r = sum (wht^T)_i*r_i
            mult = torch.sum(wht * r, dim=-1).view(-1, 1)
            psi = r - mult * r

        # Eq. (1).
        return torch.linalg.norm(h + psi - t, dim=-1, ord=self.pnorm)

    def return_score(self, is_predict=False):
        (head_emb, rel_emb, tail_emb) = self.current_batch

        h, hc = head_emb["e"], head_emb["ec"]
        t, tc = tail_emb["e"], tail_emb["ec"]
        r = rel_emb["r"]

        return self._calc(h, hc, r, t, tc, is_predict)
