import math
import torch
from Models.Model import Model


class KG2E(Model):
    """
    Shizhu He, Kang Liu, Guoliang Ji, Jun Zhao: Learning to Represent Knowledge Graphs with Gaussian Embedding. CIKM
        2015: 623-632.
    """
    def __init__(self, ent_total, rel_total, dim, cmin=.05, cmax=5, variant='kl-divergence'):
        """
            dim (int): Number of dimensions for embeddings
            cmin, cmax (float): restrictions values for covariance; cmin > 0, cmax > 0, cmax > cmin. From the paper:
                "the default configuration is as follows... (cmin, cmax)=(0.05, 5)."
            variant: either kl-divergence or expected-likelihood
        """
        super(KG2E, self).__init__(ent_total, rel_total)
        self.dim = dim
        self.cmin = cmin
        self.cmax = cmax
        self.variant = variant

        # Handy when using expected likelihood.
        self.el_constant = self.dim * torch.log(torch.tensor([2 * math.pi]))

    def get_default_loss(self):
        # Eq. (9).
        return 'margin'

    def initialize_model(self):
        self.create_embedding(self.dim, emb_type="entity", name="e_mean")
        self.create_embedding(self.dim, emb_type="relation", name="r_mean")

        # These are diagonal covariance matrixes. See Eq. (11) and line 3 in Algorithm 1.
        self.create_embedding(self.dim, emb_type="entity", name="e_cov",
                              init_method="uniform", init_params=[self.cmin, self.cmax],
                              norm_method="rescaling", norm_params={"a": self.cmin, "b": self.cmax})
        self.create_embedding(self.dim, emb_type="relation", name="r_cov",
                              init_method="uniform", init_params=[self.cmin, self.cmax],
                              norm_method="rescaling", norm_params={"a": self.cmin, "b": self.cmax})

        # Eq. (10).
        self.register_scale_constraint(emb_type="entity", name="e_mean")
        self.register_scale_constraint(emb_type="relation", name="r_mean")

    def _calc(self, h_mean, h_cov, r_mean, r_cov, t_mean, t_cov):
        # The means are subtracted and the covariances are added (see beginning of Section 3.2).
        e_mean = h_mean - t_mean
        e_cov = h_cov + t_cov
        # This is an auxiliary term that is used several times.
        re_mean = r_mean - e_mean

        if self.variant == 'kl-divergence':
            # Eq. (1).

            # Using diagonal matrices, r_cov_matrix^-1 is the matrix that multiplied by r_cov_matrix gives the identity.
            #   We just need to reverse each element.
            r_cov_inv = torch.pow(r_cov, -1)

            # trace(r_cov_matrix^-1*e_cov_matrix) = sum(pow(r_cov, -1) * e_cov) when diagonal matrices. The matrix trace
            #   is the sum of all the elements of the matrix.
            first_term = torch.sum(r_cov_inv * e_cov, dim=-1)

            # Each element in re_mean is multiplied by the corresponding element in the diagonal; then, the same element
            #   in re_mean will be multiplied and all added.
            second_term = torch.sum(torch.pow(re_mean, 2) * r_cov_inv, dim=-1)

            # The determinant of a diagonal matrix is the multiplication of all its elements. Absolute value is to avoid
            #   negatives.
            third_term = torch.log(torch.abs(torch.prod(e_cov, dim=-1)/torch.prod(r_cov, dim=-1)))

            scores = .5 * (first_term + second_term - third_term - self.dim)
        elif self.variant == 'expected-likelihood':
            # Eq. (2).

            re_cov = e_cov + r_cov

            # Same as the second term above but using re_cov^-1, which is similar to r_cov_inv above.
            first_term = torch.sum(torch.pow(re_mean, 2) * torch.pow(re_cov, -1), dim=-1)

            # Same as third term above.
            second_term = torch.log(torch.abs(torch.prod(re_cov, dim=-1)))

            scores = .5 * (first_term + second_term + self.el_constant)

        return scores

    def return_score(self, is_predict=False):
        (head_emb, rel_emb, tail_emb) = self.current_batch

        h_mean, h_cov = head_emb["e_mean"], head_emb["e_cov"]
        t_mean, t_cov = tail_emb["e_mean"], tail_emb["e_cov"]
        r_mean, r_cov = rel_emb["r_mean"], rel_emb["r_cov"]

        return self._calc(h_mean, h_cov, r_mean, r_cov, t_mean, t_cov)
