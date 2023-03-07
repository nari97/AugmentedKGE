import torch
from Models.Model import Model


class NagE(Model):
    """
    Tong Yang, Long Sha, Pengyu Hong: NagE: Non-Abelian Group Embedding for Knowledge Graphs. CIKM 2020: 1735-1742.
    """
    def __init__(self, ent_total, rel_total, dim, variant='su2'):
        """
            dim (int): Number of dimensions for embeddings
            variant can be either so3 or su2.
        """
        super(NagE, self).__init__(ent_total, rel_total)
        self.dim = dim
        self.variant = variant

    def get_default_loss(self):
        # Section 4.4.
        return 'soft_margin'

    def get_score_sign(self):
        # It is a distance (norm).
        return -1

    def initialize_model(self):
        if self.variant == 'so3':
            # Above Eq. (4).
            for component in ['x', 'y', 'z']:
                self.create_embedding(self.dim, emb_type="entity", name="e_" + component)
            for component in ['phi', 'theta', 'psi']:
                self.create_embedding(self.dim, emb_type="relation", name="r_" + component)

        if self.variant == 'su2':
            # Below Eq. (16).
            for component in ['xreal', 'ximg', 'yreal', 'yimg']:
                self.create_embedding(self.dim, emb_type="entity", name="e_" + component)
            for component in ['alpha', 'theta', 'phi']:
                self.create_embedding(self.dim, emb_type="relation", name="r_" + component)

    def _calc(self, h, r, t):
        if self.variant == 'so3':
            (t_x, t_y, t_z) = t
            (r_phi, r_theta, r_psi) = r
            (cph, cth, cps) = (torch.cos(r_phi), torch.cos(r_theta), torch.cos(r_psi))
            (sph, sth, sps) = (torch.sin(r_phi), torch.sin(r_theta), torch.sin(r_psi))
            batch_size, device = t_x.shape[0], t_x.device

            def prod(p1, p2):
                (p1_x, p1_y, p1_z) = p1
                (p2_x, p2_y, p2_z) = p2
                return p1_x * p2_x + p1_y * p2_y + p1_z * p2_z

            # Eqs. (14) and (15).
            prod_x = prod((cps*cph-cth*sps*sph, cps*sph+cth*cps*cph, sps*sth), h)
            prod_y = prod((-sps*cph-cth*sps*cph, -sps*sph+cth*cps*cph, cps*sth), h)
            prod_z = prod((cps*sth, -cps*cth, cth), h)

            components = [prod_x-t_x, prod_y-t_y, prod_z-t_z]

        if self.variant == 'su2':
            (h_xreal, h_ximg, h_yreal, h_yimg) = h
            (t_xreal, t_ximg, t_yreal, t_yimg) = t
            (r_alpha, r_theta, r_phi) = r
            batch_size, device = t_xreal.shape[0], t_xreal.device

            # Auxiliary elements.
            cos_alpha, sin_alpha = torch.cos(r_alpha), torch.sin(r_alpha)
            cos_theta, sin_theta = torch.cos(r_theta), torch.sin(r_theta)

            # Get complex numbers.
            hx = torch.view_as_complex(torch.stack((h_xreal, h_ximg), dim=-1))
            hy = torch.view_as_complex(torch.stack((h_yreal, h_yimg), dim=-1))
            tx = torch.view_as_complex(torch.stack((t_xreal, t_ximg), dim=-1))
            ty = torch.view_as_complex(torch.stack((t_yreal, t_yimg), dim=-1))

            def prod(p1, p2):
                (p1_x, p1_y) = p1
                (p2_x, p2_y) = p2
                return p1_x * p2_x + p1_y * p2_y

            # Note that elem_12 and elem_21 are the same.
            sa_times_st = sin_alpha*sin_theta
            elem_11, elem_22 = torch.view_as_complex(torch.stack((cos_alpha, sa_times_st), dim=-1)), \
                torch.view_as_complex(torch.stack((cos_alpha, -sa_times_st), dim=-1))
            elem_12 = torch.polar(torch.ones_like(r_phi), -r_phi)*sin_alpha*cos_theta

            prod_x = prod((elem_11, elem_12), (hx, hy))
            prod_y = prod((elem_12, elem_22), (hx, hy))

            components = [prod_x - tx, prod_y - ty]

        # Eq. (17).
        scores = torch.zeros(batch_size, device=device)
        for c in components:
            scores += torch.linalg.norm(c, dim=-1, ord=2)

        return scores

    def return_score(self, is_predict=False):
        (head_emb, rel_emb, tail_emb) = self.current_batch

        if self.variant == 'so3':
            h = (head_emb["e_x"], head_emb["e_y"], head_emb["e_z"])
            t = (tail_emb["e_x"], tail_emb["e_y"], tail_emb["e_z"])
            r = (rel_emb["r_phi"], rel_emb["r_theta"], rel_emb["r_psi"])

        if self.variant == 'su2':
            h = (head_emb["e_xreal"], head_emb["e_ximg"], head_emb["e_yreal"], head_emb["e_yimg"])
            t = (tail_emb["e_xreal"], tail_emb["e_ximg"], tail_emb["e_yreal"], tail_emb["e_yimg"])
            r = (rel_emb["r_alpha"], rel_emb["r_theta"], rel_emb["r_phi"])

        return self._calc(h, r, t)
