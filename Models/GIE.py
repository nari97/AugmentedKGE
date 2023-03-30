import torch
from Models.Model import Model
from Utils import PoincareUtils, GivensUtils


class GIE(Model):
    """
    Zongsheng Cao, Qianqian Xu, Zhiyong Yang, Xiaochun Cao, Qingming Huang: Geometry Interaction Knowledge Graph
        Embeddings. AAAI 2022: 5521-5529.
    """
    def __init__(self, ent_total, rel_total, dim, variant="full", apply_sigmoid=False):
        """
            dim (int): Number of dimensions for embeddings
            variant can be either full, gie1 or gie2. Note that gie3, described in the ablation study, removes the
                geometry interaction (Inter(E, H, S)); however, there are not enough details on how this is implemented.
            apply_sigmoid (Bool): Whether sigmoid must be applied to scores during training. Note that BCEWithLogitsLoss
                already applies sigmoid, so, if this is the loss function used, apply_sigmoid must be set to False. If
                a different loss function is applied, then apply_sigmoid should be set to True.
        """
        super(GIE, self).__init__(ent_total, rel_total)
        # It must be divided by two because of the R embeddings.
        self.dim = 2 * int(dim // 2)
        self.variant = variant
        self.apply_sigmoid = apply_sigmoid

    def get_default_loss(self):
        # Eq. (11).
        return 'bce'

    def get_score_sign(self):
        # It is a distance.
        return -1

    def initialize_model(self):
        self.create_embedding(self.dim, emb_type="entity", name="e")
        # Eq. (10) only contains a single bias. We use two, one for head and one for tail, as in MuRP.
        self.create_embedding(1, emb_type="entity", name="b")
        # It is not mentioned in the paper, but the implementation available here: https://github.com/Lion-ZS/GIE/ uses
        #   Givens rotation to compute R.
        # According to Eq. (6), it is divided into two vectors: r and v. We apply Givens rotation to r to get R.
        self.create_embedding(int(self.dim/2), emb_type="relation", name="r")
        self.create_embedding(self.dim, emb_type="relation", name="rv")
        # There are two attention vectors.
        self.create_embedding(self.dim, emb_type="global", name="alpha_h")
        self.create_embedding(self.dim, emb_type="global", name="alpha_t")

        # Note that c, u and v are not mentioned in the experiments in the original paper. We use a parameter per
        #   relation similar to AttH.
        self.create_embedding(1, emb_type="relation", name="c")
        self.create_embedding(1, emb_type="relation", name="u")
        self.create_embedding(1, emb_type="relation", name="v")

    # To train model in the hyperbolic, we need a special SGD (see
    #   https://github.com/ibalazevic/multirelational-poincare/blob/master/rsgd.py).
    # Instead, we optimize in the tangent space and map them to the Poincare ball. Check Section A.4 in the paper.
    def _calc(self, h, bh, alpha_h, r, rv, c, u, v, t, bt, alpha_t, is_predict):
        batch_size = h.shape[0]
        # c and v must be positive. u must be negative.
        c, u, v = torch.abs(c), -torch.abs(u), torch.abs(v)
        # To get R, we use Givens rotation multiplication using the unit vector.
        R = GivensUtils.get_rotation_matrix(r, self.dim)

        # Map r and t from Tangent to Poincare using both u an v.

        def get_inter(e, R, rv, alpha):
            # See above Eqs. (8) and (9).
            # In the appendix, one proof mentions that rh is R*h + rv.
            E = torch.bmm(R, e.view(batch_size, -1, 1)).view(batch_size, -1) + rv
            e_u, e_v = PoincareUtils.exp_map(e, u), PoincareUtils.exp_map(e, v)
            # In the appendix, one proof mentions that rh is R*h + rv.
            H = PoincareUtils.mobius_matrix_multiplication(R, e_v, v) + rv
            S = PoincareUtils.mobius_matrix_multiplication(R, e_u, u) + rv
            a_e = torch.nn.functional.softmax(torch.bmm(alpha.expand(batch_size, -1).view(batch_size, 1, -1),
                                                        E.view(batch_size, -1, 1)).view(batch_size), dim=0)
            a_h = torch.nn.functional.softmax(torch.bmm(alpha.expand(batch_size, -1).view(batch_size, 1, -1),
                                                        H.view(batch_size, -1, 1)).view(batch_size), dim=0)
            a_s = torch.nn.functional.softmax(torch.bmm(alpha.expand(batch_size, -1).view(batch_size, 1, -1),
                                                        S.view(batch_size, -1, 1)).view(batch_size), dim=0)

            # Eqs. (8) and (9).
            return PoincareUtils.exp_map(a_e.view(-1, 1) * E + a_h.view(-1, 1) * PoincareUtils.log_map(H, v) +
                                         a_s.view(-1, 1) * PoincareUtils.log_map(S, u), c)

        if self.variant == 'gie1' or self.variant == 'full':
            dch = PoincareUtils.geodesic_dist(get_inter(h, R, rv, alpha_h), t, c)
        else:
            dch = torch.zeros_like(bh)

        # Footnote 2 says R = R^T and v = -R^T * v
        RT = torch.transpose(R, dim0=1, dim1=2)
        RTv = torch.bmm(RT, rv.view(batch_size, -1, 1)).view(batch_size, -1)
        if self.variant == 'gie2' or self.variant == 'full':
            dct = PoincareUtils.geodesic_dist(get_inter(t, RT, -RTv, alpha_t), h, c)
        else:
            dct = torch.zeros_like(bt)

        # Eq. (10).
        scores = dch - dct - bh - bt

        # Apply sigmoid when predicting or when indicated by apply_sigmoid.
        if is_predict or self.apply_sigmoid:
            scores = torch.sigmoid(scores)

        return scores

    def return_score(self, is_predict=False):
        (head_emb, rel_emb, tail_emb) = self.current_batch

        h, bh = head_emb["e"], head_emb["b"]
        t, bt = tail_emb["e"], tail_emb["b"]
        r, rv, c, u, v = rel_emb["r"], rel_emb["rv"], rel_emb["c"], rel_emb["u"], rel_emb["v"]
        alpha_h, alpha_t = self.get_global_embeddings()['alpha_h'], self.get_global_embeddings()['alpha_t']

        return self._calc(h, bh, alpha_h, r, rv, c, u, v, t, bt, alpha_t, is_predict)
