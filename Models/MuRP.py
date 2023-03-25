import torch
from Models.Model import Model
from Utils import PoincareUtils


# Check this: https://github.com/ibalazevic/multirelational-poincare/blob/master/model.py
class MuRP(Model):
    """
    Ivana Balazevic, Carl Allen, Timothy M. Hospedales: Multi-relational Poincaré Graph Embeddings. NeurIPS 2019:
        4465-4475.
    """
    def __init__(self, ent_total, rel_total, dim, variant="murp", apply_sigmoid=False):
        """
            dim (int): Number of dimensions for embeddings
            variant can be either murp or mure.
            apply_sigmoid (Bool): Whether sigmoid must be applied to scores during training. Note that BCEWithLogitsLoss
                already applies sigmoid, so, if this is the loss function used, apply_sigmoid must be set to False. If
                a different loss function is applied, then apply_sigmoid should be set to True.
        """
        super(MuRP, self).__init__(ent_total, rel_total)
        self.dim = dim
        self.variant = variant
        self.apply_sigmoid = apply_sigmoid

    def get_default_loss(self):
        # Eq. (6)
        return 'bce'

    def get_score_sign(self):
        # It is a distance.
        return -1

    def initialize_model(self):
        # From the paper: "We initialize all embeddings near the origin where distances are small in hyperbolic space,
        #   similar to [25]."
        # From [25]: "First, we initialize all embeddings randomly from the uniform distribution U(−0.001, 0.001)."
        # When initializing using this option, the model does not learn properly; we do not use it.
        self.create_embedding(self.dim, emb_type="entity", name="e"
                              #, init_method="uniform", init_params=[-.001, .001]
                              )
        self.create_embedding(1, emb_type="entity", name="b"
                              #, init_method="uniform", init_params=[-.001, .001]
                              )

        self.create_embedding(self.dim, emb_type="relation", name="r"
                              #, init_method="uniform", init_params=[-.001, .001]
                              )
        self.create_embedding(self.dim, emb_type="relation", name="R",
                              #init_method="uniform", init_params=[-.001, .001]
                              )

        # Only when using Poincare.
        # Note that c=1 in the original paper; however, we use a parameter per relation similar to AttH.
        if self.variant.endswith('p'):
            self.create_embedding(1, emb_type="relation", name="c")

    # To train model in the hyperbolic, we need a special SGD (see
    #   https://github.com/ibalazevic/multirelational-poincare/blob/master/rsgd.py).
    # Instead, we optimize in the tangent space and map them to the Poincare ball. Check Section A.4 in the paper.
    def _calc(self, h, bh, r, R, c, t, bt, is_predict):
        if self.variant.endswith('p'):
            # c must be positive and is only present when dealing with Poincare.
            c = torch.abs(c)
            # Map r and t from Tangent to Poincare.
            r, t = PoincareUtils.exp_map(r, c), PoincareUtils.exp_map(t, c)

        # Note that h_s in Eq. (5) is hyperbolic and is transformed into tangent using log_map; however, h in our case
        #   is already in the tangent space, so we use it directly.
        # Eq. (4) and (5).
        r_times_h = R*h

        if self.variant.endswith('p'):
            # Eq. (5). Map to Poincare.
            r_times_h = PoincareUtils.exp_map(r_times_h, c)

        if self.variant.endswith('p'):
            # Eq. (5).
            t_plus_r = PoincareUtils.mobius_addition(t, r, c)
        else:
            # Eq. (4).
            t_plus_r = t + r

        if self.variant.endswith('p'):
            # Eq. (5) with different sign.
            scores = PoincareUtils.geodesic_dist(r_times_h, t_plus_r, c) ** 2 - bh - bt
        else:
            # Eq. (4). Just simple Euclidean norm.
            scores = torch.linalg.norm(r_times_h - t_plus_r, dim=-1, ord=2) ** 2 - bh.flatten() - bt.flatten()

        # Apply sigmoid when predicting or when indicated by apply_sigmoid.
        if is_predict or self.apply_sigmoid:
            scores = torch.sigmoid(scores)

        return scores

    def return_score(self, is_predict=False):
        (head_emb, rel_emb, tail_emb) = self.current_batch

        h, bh = head_emb["e"], head_emb["b"]
        t, bt = tail_emb["e"], tail_emb["b"]
        r, R, c = rel_emb["r"], rel_emb["R"], rel_emb.get("c", None)

        return self._calc(h, bh, r, R, c, t, bt, is_predict)
