import math
import torch
from Models.Model import Model


class ITransF(Model):
    """
    Qizhe Xie, Xuezhe Ma, Zihang Dai, Eduard H. Hovy: An Interpretable Knowledge Transfer Model for Knowledge Base
        Completion. ACL (1) 2017: 950-962.
    """
    def __init__(self, ent_total, rel_total, dim, norm=2, relation_reduction=.01):
        """
            dim (int): Number of dimensions for embeddings
            norm (int): L1 or L2 norm. Default: 2
            relation_reduction (float): Between 0 and 1, it indicates the value of m (concept matrices), which is a
                percentage over the total number of relations. Default: .01.
        """
        super(ITransF, self).__init__(ent_total, rel_total)
        self.dim = dim
        self.pnorm = norm
        # We use at least m=2.
        self.m = max(2, math.ceil(rel_total * relation_reduction))

    def get_default_loss(self):
        # Eq. (2).
        return 'margin'

    def get_score_sign(self):
        # It is a distance (norm).
        return -1

    def initialize_model(self):
        # See Eq. (1).
        # "We normalize the entity vectors h, t."
        self.create_embedding(self.dim, emb_type="entity", name="e", norm_method="norm")
        self.create_embedding(self.dim, emb_type="relation", name="r")
        # Matrix D. "We initialize the projection matrices with identity matrices added with a small noise sampled from
        #   normal distribution N(0, 0.005^2)."
        self.create_embedding((self.m, self.dim,self.dim), emb_type="global", name="d",
                              init_method="uniform", init_params=[1, 1])
        d_emb = self.get_embedding("global", "d").emb
        d_emb.data = d_emb.data + torch.zeros_like(d_emb).normal_(mean=0, std=.005**2)

        # Attention mechanism (see "Sparse attention vectors" paragraph).
        self.create_embedding(self.m, emb_type="relation", name="vrh")
        self.create_embedding(self.m, emb_type="relation", name="vrt")
        self.create_embedding(self.m, emb_type="relation", name="ih", init_method="uniform", init_params=[0, 1])
        self.create_embedding(self.m, emb_type="relation", name="it", init_method="uniform", init_params=[0, 1])
        # Temperature of softmax. It is a hyperparameter in the paper; we use a parameter.
        self.create_embedding(1, emb_type="global", name="tau", init_method="uniform", init_params=[0, 1])

    def _calc(self, h, r, vrh, vrt, ih, it, t, d, tau, is_predict):
        # d must be replicated until batch_size.
        batch_size = h.shape[0]
        d = d.expand((batch_size, -1, -1, -1))

        # See the SparseSoftmax function.
        def sparsesoftmax(v, i):
            # Min-max scaling over i to be between 0 and 1.
            min, max = torch.min(i), torch.max(i)
            i = (i - min)/(max - min)
            # The values must be either 0 or 1.
            i_ones = i >= .5
            i = torch.where(i_ones, 1, 0)
            # This is not exactly the same implementation as in the paper (i is multiplied after exp(v/tau)); however,
            #   that implementation does not work!
            return torch.nn.functional.softmax((v*i)/tau, dim=1)

        ah, at = sparsesoftmax(vrh, ih), sparsesoftmax(vrt, it)
        # When using bmm, both tensors must have three dimensions: (batch_size, first_dim, second_dim).
        d = d.view(batch_size, self.m, -1)

        def get_transfer(a, e):
            a_times_d = torch.bmm(a.view(batch_size, 1, self.m), d)
            # Change back to original dimensions.
            a_times_d = a_times_d.view(batch_size, self.dim, self.dim)
            # Multiply by entity vector and change to original dimensions.
            return torch.bmm(a_times_d, e.view(batch_size, self.dim, 1)).view(batch_size, self.dim)

        # These are the multiplications by D and attention vectors.
        h_transfer, t_transfer = get_transfer(ah, h), get_transfer(at, t)

        # "We normalize the projected entity vectors..." That is, the transfers. We use the same as TransD.
        if not is_predict:
            self.onthefly_constraints.append(self.scale_constraint(h_transfer))
            self.onthefly_constraints.append(self.scale_constraint(t_transfer))

        # Eq. (1).
        return torch.linalg.norm(h_transfer + r - t_transfer, dim=-1, ord=self.pnorm)

    def return_score(self, is_predict=False):
        (head_emb, rel_emb, tail_emb) = self.current_batch

        h = head_emb["e"]
        t = tail_emb["e"]
        r, vrh, vrt, ih, it = rel_emb["r"], rel_emb["vrh"], rel_emb["vrt"], rel_emb["ih"], rel_emb["it"]
        d, tau = self.current_global_embeddings["d"], self.current_global_embeddings["tau"]

        return self._calc(h, r, vrh, vrt, ih, it, t, d, tau, is_predict)
