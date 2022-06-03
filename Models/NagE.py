import math
import torch
from Models.Model import Model


class NagE(Model):
    def __init__(self, ent_total, rel_total, dim):
        super(NagE, self).__init__(ent_total, rel_total)
        self.dim = dim

    def get_default_loss(self):
        return 'soft_margin'

    def initialize_model(self):
        for component in ['x', 'y', 'z']:
            self.create_embedding(self.dim, emb_type="entity", name="e_" + component)
        for component in ['phi', 'theta', 'psi']:
            self.create_embedding(self.dim, emb_type="relation", name="r_" + component,
                                  init="uniform", init_params=[0, 2 * math.pi])

    def _calc(self, h, r, t):
        (t_x, t_y, t_z) = t
        (r_phi, r_theta, r_psi) = r
        (cph, cth, cps) = (torch.cos(r_phi), torch.cos(r_theta), torch.cos(r_psi))
        (sph, sth, sps) = (torch.sin(r_phi), torch.sin(r_theta), torch.sin(r_psi))

        batch_size = t_x.shape[0]

        def prod(p1, p2):
            (p1_x, p1_y, p1_z) = p1
            (p2_x, p2_y, p2_z) = p2
            return p1_x*p2_x + p1_y*p2_y + p1_z*p2_z

        # Check Eq. 15.
        prod_x = prod((cps*cph-cth*sps*sph, cps*sph+cth*cps*cph, sps*sth), h)
        prod_y = prod((-sps*cph-cth*sps*cph, -sps*sph+cth*cps*cph, cps*sth), h)
        prod_z = prod((cps*sth, -cps*cth, cth), h)

        # We add all components.
        scores = torch.zeros(batch_size, device=t_x.device)
        for c in [prod_x-t_x, prod_y-t_y, prod_z-t_z]:
            scores += torch.linalg.norm(c, dim=-1, ord=2)

        return -scores

    def return_score(self, is_predict=False):
        (head_emb, rel_emb, tail_emb) = self.current_batch

        (h_x, h_y, h_z) = (head_emb["e_x"], head_emb["e_y"], head_emb["e_z"])
        (t_x, t_y, t_z) = (tail_emb["e_x"], tail_emb["e_y"], tail_emb["e_z"])
        (r_phi, r_theta, r_psi) = (rel_emb["r_phi"], rel_emb["r_theta"], rel_emb["r_psi"])

        return self._calc((h_x, h_y, h_z), (r_phi, r_theta, r_psi), (t_x, t_y, t_z))
