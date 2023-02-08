import torch
from Models.Model import Model


class BoxE(Model):
    """
    Ralph Abboud, Ismail Ilkan Ceylan, Thomas Lukasiewicz, Tommaso Salvatori: BoxE: A Box Embedding Model for Knowledge
        Base Completion. NeurIPS 2020.
    """
    def __init__(self, ent_total, rel_total, dim, norm=2):
        """
            dim (int): Number of dimensions for embeddings
            norm (int): L1 or L2 norm. Default: 2
        """
        super(BoxE, self).__init__(ent_total, rel_total)
        self.dim = dim
        self.pnorm = norm

    def get_default_loss(self):
        # Negative sampling loss (reference to RotatE).
        return 'soft_margin'

    def initialize_model(self):
        # From the paper: "every entity is represented by two vectors e and b, where e defines the base position of the
        #   entity, and b defines its translational bump, which translates all the entities co-occuring in a fact with
        #   e, from their base positions to their final embeddings by "bumping" them."
        self.create_embedding(self.dim, emb_type="entity", name="e_base")
        # Not proposed in the original paper; the implementation
        #   (https://github.com/ralphabb/BoxE/blob/master/BoxEModel.py#L398) normalizes only when initializing.
        #   We empirically found that this norm helps.
        self.create_embedding(self.dim, emb_type="entity", name="e_bump", norm_method="norm")

        # From the paper: "every relation r is represented by n hyper-rectangles, i.e., boxes, r(1),...,r(n), where n
        #   is the arity of r. Intuitively, this representation defines n regions, one per arity position, such that
        #   a fact r(e1,...,en) holds when the final embeddings of e1,...,en each appear in their corresponding
        #   position box, creating a class abstraction for the sets of all entities appearing at every arity position."
        # A box has a base and a delta, check: https://github.com/ralphabb/BoxE/blob/master/BoxEModel.py#L119.
        self.create_embedding(self.dim, emb_type="relation", name="r_base_1")
        self.create_embedding(self.dim, emb_type="relation", name="r_delta_1")
        self.create_embedding(self.dim, emb_type="relation", name="r_base_2")
        self.create_embedding(self.dim, emb_type="relation", name="r_delta_2")

    # This method computes lower and upper boundaries, center points and widths plus one:
    #   https://github.com/ralphabb/BoxE/blob/master/BoxEModel.py#L119
    def get_box(self, base, delta):
        box_second = base + .5 * delta
        box_first = base - .5 * delta
        box_lower = torch.minimum(box_first, box_second)
        box_upper = torch.maximum(box_first, box_second)
        return box_lower, box_upper, (box_upper + box_lower) / 2, box_upper - box_lower + 1

    # See dist function in the paper.
    def dist(self, base, bump, low, high, center, width):
        # Check the points that are within box; within_box[i]=True means the point is within the box.
        within_box = (torch.min(low <= base, dim=1) and torch.min(base <= high, dim=1))[0] == True
        # This is common in both cases in the function.
        common = torch.abs(base + bump) - center
        # For those within box, divide by width; otherwise, do nothing.
        common = torch.where(within_box.view(-1, 1), common/width, common)
        # For those outside box, multiply by width and minus kappa; otherwise, just do nothing.
        common = torch.where(~within_box.view(-1, 1), common * width - .5*(width-1)*(width-1/width), common)
        return common

    def _calc(self, hbase, hbump, r_base_1, r_delta_1, r_base_2, r_delta_2, tbase, tbump):
        # Get boxes, center points and widths.
        box_lower_1, box_higher_1, box_center_1, box_width_1 = self.get_box(r_base_1, r_delta_1)
        box_lower_2, box_higher_2, box_center_2, box_width_2 = self.get_box(r_base_2, r_delta_2)

        # Compute distances.
        dist_1 = self.dist(hbase, tbump, box_lower_1, box_higher_1, box_center_1, box_width_1)
        dist_2 = self.dist(tbase, hbump, box_lower_2, box_higher_2, box_center_2, box_width_2)

        # Apply norms (see score function).
        return -torch.linalg.norm(dist_1, dim=-1, ord=self.pnorm) -torch.linalg.norm(dist_2, dim=-1, ord=self.pnorm)

    def return_score(self, is_predict=False):
        (head_emb, rel_emb, tail_emb) = self.current_batch

        # From the paper: "For all BoxE experiments, points and boxes were projected into the hypercube [-1, 1], a
        #   bounded space, by simply applying the hyperbolic tangent function tanh element-wise on all final
        #   embedding representations."
        hbase, hbump = torch.tanh(head_emb["e_base"]), torch.tanh(head_emb["e_bump"])
        tbase, tbump = torch.tanh(tail_emb["e_base"]), torch.tanh(tail_emb["e_bump"])
        r_base_1, r_delta_1 = torch.tanh(rel_emb["r_base_1"]), torch.tanh(rel_emb["r_delta_1"])
        r_base_2, r_delta_2 = torch.tanh(rel_emb["r_base_2"]), torch.tanh(rel_emb["r_delta_2"])

        return self._calc(hbase, hbump, r_base_1, r_delta_1, r_base_2, r_delta_2, tbase, tbump)
