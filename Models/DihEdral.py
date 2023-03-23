import torch
import numpy as np
from Models.Model import Model


class DihEdral(Model):
    """
    Canran Xu, Ruijiang Li: Relation Embedding with Dihedral Group in Knowledge Graph. ACL (1) 2019: 263-272.
    The paper suggests to implementions: Gumbel-Softmax and binary variables. We implement the former only.
    """
    def __init__(self, ent_total, rel_total, dim, variant='4'):
        """
            dim (int): Number of dimensions for embeddings
            variant can be either 4, 6 or 8, which is the value of K.
        """
        super(DihEdral, self).__init__(ent_total, rel_total)
        # We cannot use odd dimensions as we are going to partition embeddings into chunks of size 2.
        self.dim = 2*(dim//2)
        self.K = int(variant)
        self.Dk = []

    def get_default_loss(self):
        # Eq. (2).
        return 'soft'

    def get_score_sign(self):
        # It is a similarity.
        return 1

    def initialize_model(self):
        # See Eq. (2) for regularization.
        self.create_embedding(self.dim, emb_type="entity", name="e", reg=True)
        # The size is self.dim*K (See Section 4.1).
        self.create_embedding(self.dim*self.K, emb_type="relation", name="r")

        # Compute Dk (see Eq. (1)). We compute each rotation and reflection matrices.
        for m in range(self.K):
            ok = self.create_embedding((2, 2), emb_type="global", register=False)
            ok.requires_grad = False
            ok.emb.data[0,0,0] = np.cos(2 * np.pi * m / self.K)
            ok.emb.data[0,0,1] = -np.sin(2 * np.pi * m / self.K)
            ok.emb.data[0,1,0] = np.sin(2 * np.pi * m / self.K)
            ok.emb.data[0,1,1] = np.cos(2 * np.pi * m / self.K)
            self.Dk.append(ok)

            fk = self.create_embedding((2, 2), emb_type="global", register=False)
            fk.requires_grad = False
            fk.emb.data[0,0,0] = np.cos(2 * np.pi * m / self.K)
            fk.emb.data[0,0,1] = np.sin(2 * np.pi * m / self.K)
            fk.emb.data[0,1,0] = np.sin(2 * np.pi * m / self.K)
            fk.emb.data[0,1,1] = -np.cos(2 * np.pi * m / self.K)
            self.Dk.append(fk)
        
    def _calc(self, h, r, t):
        batch_size = h.shape[0]
        scores = torch.zeros((batch_size, 1), dtype=h.dtype, layout=h.layout, device=h.device)

        # See Section 3. This is the l=1..L summation.
        for l in range(0, self.dim, 2):
            h_slice, t_slice = h[:, l:l+2], t[:, l:l+2]
            s = r[:, l*self.K:(l+2)*self.K]

            # Take 2K random samples from U(0, 1) and apply -log(-log) to them.
            q = torch.zeros_like(s)
            torch.rand((batch_size, 2 * self.K), out=q)
            q = -torch.log(-torch.log(q))

            # Temperature: starts in 3; then, max(0.5, 3*exp(−0.001*epoch)).
            if self.epoch == 0:
                tau = 3
            else:
                tau = max(.5, 3*np.exp(-.001*self.epoch))
            c = torch.nn.functional.softmax((s+q)/tau, dim=1)

            # We take each matrix in Dk, multiply by c and add to r_slice.
            r_slice = torch.zeros((batch_size, 2, 2), dtype=h.dtype, layout=h.layout, device=h.device)
            for k in range(self.K*2):
                r_slice += c[:,k].view(-1, 1, 1) * self.Dk[k].emb.expand(batch_size, 2, 2)

            # See Section 3.
            scores += torch.sum(
                torch.bmm(h_slice.view(batch_size, 1, 2), r_slice).view(batch_size, 2) * t_slice, -1).view(-1, 1)

        return scores

    def return_score(self, is_predict=False):
        (head_emb, rel_emb, tail_emb) = self.current_batch

        h = head_emb["e"]
        t = tail_emb["e"]
        r = rel_emb["r"]

        return self._calc(h, r, t)
