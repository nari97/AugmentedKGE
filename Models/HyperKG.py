import math
import torch
from Models.Model import Model
from Utils import PoincareUtils


class HyperKG(Model):
    """
    Prodromos Kolyvakis, Alexandros Kalousis, Dimitris Kiritsis: Hyperbolic Knowledge Graph Embeddings for Knowledge
        Base Completion. ESWC 2020: 199-214.
    """
    def __init__(self, ent_total, rel_total, dim, beta=None, variant="euclidean"):
        """
            dim (int): Number of dimensions for embeddings
            variant can be either euclidean or mobius.
        """
        super(HyperKG, self).__init__(ent_total, rel_total)
        self.dim = dim
        if beta is None:
            # In the experiments, beta={3*dim/4, dim/2, dim/4, 0}. In the supplemental, dim/2 was consistently the best.
            self.beta = math.floor(self.dim/2)
        else:
            self.beta = beta
        self.variant = variant

    def get_default_loss(self):
        # Eq. (7).
        # There is a regularization term R (see Eq. (8)) that we do not implement.
        return 'margin'

    def get_score_sign(self):
        # It is a distance.
        return -1

    def initialize_model(self):
        # From the paper: "We initialise the embeddings using the Xavier initialization scheme."
        self.create_embedding(self.dim, emb_type="entity", name="e")
        self.create_embedding(self.dim, emb_type="relation", name="r")
        # Note that c is not mentioned in the original paper; however, we use a parameter per relation similar to AttH.
        self.create_embedding(1, emb_type="relation", name="c")

        # From the paper: "To enforce the term embeddings to stay in the Poincare-ball, we constrain all the entity
        #   embeddings to have a Euclidean norm less than 0.5. Namely, ||e|| < 0.5 and ||r|| < 1.0 for all entity and
        #   relation vectors, respectively." See also Eq. (9).
        # From the paper: "In the experiment where the Mobius addition was used, we removed the constraint for the
        #   entity vectors to have a norm less than 0.5."
        if self.variant == "euclidean":
            self.register_scale_constraint(emb_type="entity", name="e", z=.5)
        self.register_scale_constraint(emb_type="relation", name="r")

    # To train model in the hyperbolic, we need a special SGD (see
    #   https://github.com/ibalazevic/multirelational-poincare/blob/master/rsgd.py).
    # Instead, we optimize in the tangent space and map them to the Poincare ball. Check Section A.4 in the paper.
    def _calc(self, h, r, c, t):
        # We want to rotate t beta times.
        t = torch.roll(t, shifts=self.beta, dims=1)
        # c must be positive and is only present when dealing with Poincare.
        c = torch.abs(c)
        # Map r from Tangent to Poincare.
        r = PoincareUtils.exp_map(r, c)

        if self.variant == "euclidean":
            # Map from Tangent to Poincare.
            h_plus_t = PoincareUtils.exp_map(h+t, c)
        if self.variant == "mobius":
            # Map h and t from Tangent to Poincare.
            h, t = PoincareUtils.exp_map(h, c), PoincareUtils.exp_map(t, c)
            h_plus_t = PoincareUtils.mobius_addition(h, t, c)
        # Eq. (6).
        return PoincareUtils.distance(h_plus_t, r)

    def return_score(self, is_predict=False):
        (head_emb, rel_emb, tail_emb) = self.current_batch

        h = head_emb["e"]
        t = tail_emb["e"]
        r, c = rel_emb["r"], rel_emb["c"]

        return self._calc(h, r, c, t)
