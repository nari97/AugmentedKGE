import math
import torch
from Models.Model import Model


class KG2E(Model):
    # variant is either kl-divergence or expected-likelihood
    def __init__(self, ent_total, rel_total, dim, cmin=.05, cmax=5, variant='kl-divergence'):
        super(KG2E, self).__init__(ent_total, rel_total)
        self.dim = dim
        self.cmin = cmin
        self.cmax = cmax
        self.variant = variant

    def get_default_loss(self):
        return 'margin'

    def initialize_model(self):
        self.create_embedding(self.dim, emb_type="entity", name="e_mean")
        # This is a diagonal covariance matrix.
        self.create_embedding(self.dim, emb_type="entity", name="e_cov",
                              init="uniform", init_params=[self.cmin, self.cmax],
                              norm_method="rescaling", norm_params={"a": self.cmin, "b": self.cmax})

        self.create_embedding(self.dim, emb_type="relation", name="r_mean")
        self.create_embedding(self.dim, emb_type="relation", name="r_cov",
                              init="uniform", init_params=[self.cmin, self.cmax],
                              norm_method="rescaling", norm_params={"a": self.cmin, "b": self.cmax})

        self.register_scale_constraint(emb_type="entity", name="e_mean")
        self.register_scale_constraint(emb_type="relation", name="r_mean")

    def get_cov_matrix(self, v, batch_size):
        # v = [a, b, c]
        # We wish to return:
        #   a, 0, 0
        #   0, b, 0
        #   0, 0, c
        return torch.diag_embed(v)

    def _calc(self, h_mean, h_cov, r_mean, r_cov, t_mean, t_cov):
        batch_size = h_mean.shape[0]

        # Get covariance matrices.
        h_cm = self.get_cov_matrix(h_cov, batch_size)
        t_cm = self.get_cov_matrix(t_cov, batch_size)
        r_cm = self.get_cov_matrix(r_cov, batch_size)

        e_mean = h_mean - t_mean
        e_cm = h_cm + t_cm

        re_mean = r_mean - e_mean
        if self.variant == 'kl-divergence':
            r_icm = torch.linalg.inv(r_cm)

            scores = .5 * (torch.diagonal(torch.bmm(r_icm, e_cm), dim1=-2, dim2=-1).sum(-1) + \
                torch.bmm(torch.bmm(re_mean.view(batch_size, 1, self.dim), r_icm),
                          re_mean.view(batch_size, self.dim, 1)).flatten() \
                    # Included absolute value to avoid negative values.
                    - torch.log(torch.abs(torch.linalg.det(e_cm)/torch.linalg.det(r_cm))) - self.dim)
        elif self.variant == 'expected-likelihood':
            re_cm = e_cm + r_cm

            scores = .5 * (torch.bmm(torch.bmm(re_mean.view(batch_size, 1, self.dim), torch.linalg.inv(re_cm)),
                                     re_mean.view(batch_size, self.dim, 1)).flatten() +
                           # Included absolute value to avoid negative values?
                           torch.log(torch.abs(torch.linalg.det(re_cm))) +
                           self.dim * torch.log(torch.tensor([2*math.pi])))

        return scores

    def return_score(self, is_predict=False):
        (head_emb, rel_emb, tail_emb) = self.current_batch

        h_mean, h_cov = head_emb["e_mean"], head_emb["e_cov"]
        t_mean, t_cov = tail_emb["e_mean"], tail_emb["e_cov"]
        r_mean, r_cov = rel_emb["r_mean"], rel_emb["r_cov"]

        return self._calc(h_mean, h_cov, r_mean, r_cov, t_mean, t_cov)
