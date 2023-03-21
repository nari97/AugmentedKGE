import torch
from Models.Model import Model


class GeomE(Model):
    """
    Chengjin Xu, Mojtaba Nayyeri, Yung-Yu Chen, Jens Lehmann: Knowledge Graph Embeddings in Geometric Algebras. COLING
        2020: 530-544.
    """
    def __init__(self, ent_total, rel_total, dim, variant='2d'):
        """
            dim (int): Number of dimensions for embeddings
            variant can be either 2d, 3d or plus.
        """
        super(GeomE, self).__init__(ent_total, rel_total)
        self.dim = dim
        self.variant = variant

    def get_default_loss(self):
        # Eq. (6). They differentiate between head and tail corruptions. We do not do that.
        return 'logsoftmax'

    def get_score_sign(self):
        # It is a similarity.
        return 1

    def initialize_model(self):
        # If 2D, there are four components; if 3D, there are eight components.
        # All parameters are regularized, see Eq. (6).
        if self.variant == '2d' or self.variant == 'plus':
            for x in ['0', '1', '2', '12']:
                self.create_embedding(self.dim, emb_type="entity", name="e2d_"+x, reg=True)
                self.create_embedding(self.dim, emb_type="relation", name="r2d_"+x, reg=True)

        if self.variant == '3d' or self.variant == 'plus':
            for x in ['0', '1', '2', '3', '12', '13', '23', '123']:
                self.create_embedding(self.dim, emb_type="entity", name="e3d_"+x, reg=True)
                self.create_embedding(self.dim, emb_type="relation", name="r3d_"+x, reg=True)

    def _calc_2d(self, h, r, t):
        (h_0, h_1, h_2, h_12) = h
        (r_0, r_1, r_2, r_12) = r
        (t_0, t_1, t_2, t_12) = t
        # Eq. (13).
        return torch.sum((h_0 * r_0 + h_1 * r_1 + h_2 * r_2 - h_12 * r_12)*t_0
                         - (h_0 * r_1 + h_1 * r_0 - h_2 * r_12 + h_12 * r_2)*t_1
                         - (h_0 * r_2 + h_2 * r_0 + h_1 * r_12 - h_12 * r_1)*t_2
                         + (h_1 * r_2 - h_2 * r_1 + h_0 * r_12 + h_12 * r_0)*t_12, -1)

    def _calc_3d(self, h, r, t):
        (h_0, h_1, h_2, h_3, h_12, h_13, h_23, h_123) = h
        (r_0, r_1, r_2, r_3, r_12, r_13, r_23, r_123) = r
        (t_0, t_1, t_2, t_3, t_12, t_13, t_23, t_123) = t
        # Eq. (14).
        return torch.sum((h_0*r_0+h_1*r_1+h_2*r_2+h_3*r_3-h_12*r_12-h_23*r_23-h_13*r_13-h_123*r_123)*t_0
                         - (h_0*r_1+h_1*r_0-h_2*r_12+h_12*r_2-h_3*r_13+h_13*r_3-h_23*r_123-h_123*r_23)*t_1
                         - (h_0*r_2+h_2*r_0+h_1*r_12-h_12*r_1-h_3*r_23+h_23*r_3+h_13*r_123+h_123*r_13)*t_2
                         - (h_0*r_3+h_3*r_0+h_1*r_13-h_13*r_1+h_2*r_23-h_23*r_2-h_12*r_123-h_123*r_12)*t_3
                         + (h_0*r_12+h_12*r_0+h_1*r_2-h_2*r_1-h_13*r_23+h_23*r_13+h_3*r_123+h_123*r_3)*t_12
                         + (h_0*r_23+h_23*r_0+h_1*r_123+h_123*r_1+h_2*r_3-h_3*r_2-h_12*r_13+h_13*r_12)*t_23
                         + (h_0*r_13+h_13*r_0+h_1*r_3-h_3*r_1-h_2*r_123-h_123*r_2+h_12*r_23-h_23*r_12)*t_13
                         + (h_0*r_123+h_123*r_0+h_1*r_23+h_23*r_1-h_2*r_13-h_13*r_2+h_3*r_12+h_12*r_3)*t_123, -1)

    def return_score(self, is_predict=False):
        (head_emb, rel_emb, tail_emb) = self.current_batch

        if self.variant == '2d' or self.variant == 'plus':
            h_0, h_1, h_2, h_12 = head_emb["e2d_0"], head_emb["e2d_1"], head_emb["e2d_2"], head_emb["e2d_12"]
            t_0, t_1, t_2, t_12 = tail_emb["e2d_0"], tail_emb["e2d_1"], tail_emb["e2d_2"], tail_emb["e2d_12"]
            r_0, r_1, r_2, r_12 = rel_emb["r2d_0"], rel_emb["r2d_1"], rel_emb["r2d_2"], rel_emb["r2d_12"]
            scores_2d = self._calc_2d((h_0, h_1, h_2, h_12), (r_0, r_1, r_2, r_12), (t_0, t_1, t_2, t_12))

        if self.variant == '3d' or self.variant == 'plus':
            h_0, h_1, h_2, h_3, h_12, h_13, h_23, h_123 = head_emb["e3d_0"], head_emb["e3d_1"], head_emb["e3d_2"], \
                head_emb["e3d_3"], head_emb["e3d_12"], head_emb["e3d_13"], head_emb["e3d_23"], head_emb["e3d_123"]
            t_0, t_1, t_2, t_3, t_12, t_13, t_23, t_123 = tail_emb["e3d_0"], tail_emb["e3d_1"], tail_emb["e3d_2"], \
                tail_emb["e3d_3"], tail_emb["e3d_12"], tail_emb["e3d_13"], tail_emb["e3d_23"], tail_emb["e3d_123"]
            r_0, r_1, r_2, r_3, r_12, r_13, r_23, r_123 = rel_emb["r3d_0"], rel_emb["r3d_1"], rel_emb["r3d_2"], \
                rel_emb["r3d_3"], rel_emb["r3d_12"], rel_emb["r3d_13"], rel_emb["r3d_23"], rel_emb["r3d_123"]
            scores_3d = self._calc_3d((h_0, h_1, h_2, h_3, h_12, h_13, h_23, h_123),
                                      (r_0, r_1, r_2, r_3, r_12, r_13, r_23, r_123),
                                      (t_0, t_1, t_2, t_3, t_12, t_13, t_23, t_123))

        if self.variant == '2d':
            return scores_2d
        elif self.variant == '3d':
            return scores_3d
        elif self.variant == 'plus':
            return scores_2d + scores_3d
