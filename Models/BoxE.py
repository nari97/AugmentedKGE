import torch
from Models.Model import Model


class BoxE(Model):

    def __init__(self, ent_total, rel_total, dim, norm=2):
        super(BoxE, self).__init__(ent_total, rel_total)
        self.dim = dim
        self.pnorm = norm

    def get_default_loss(self):
        return 'soft_margin'

    def initialize_model(self):
        self.create_embedding(self.dim, emb_type="entity", name="e_base")
        # Not proposed in the original paper; the implementation
        #   (https://github.com/ralphabb/BoxE/blob/master/BoxEModel.py#L398) normalizes only when initializing.
        #   We empirically found that this norm helps.
        self.create_embedding(self.dim, emb_type="entity", name="e_bump", norm_method="norm")

        # A box has a base and a delta, check: https://github.com/ralphabb/BoxE/blob/master/BoxEModel.py#L119
        self.create_embedding(self.dim, emb_type="relation", name="r_base_1")
        self.create_embedding(self.dim, emb_type="relation", name="r_delta_1")
        self.create_embedding(self.dim, emb_type="relation", name="r_base_2")
        self.create_embedding(self.dim, emb_type="relation", name="r_delta_2")

    def get_box(self, base, delta):
        box_second = base + .5 * delta
        box_first = base - .5 * delta
        box_low = torch.minimum(box_first, box_second)
        box_high = torch.maximum(box_first, box_second)
        return box_low, box_high, (box_high + box_low) / 2, box_high - box_low + 1

    def dist(self, base, bump, low, high, center, width):
        within_box = (torch.min(low <= base, dim=1) and torch.min(base <= high, dim=1))[0] == True
        common = torch.abs(base + bump) - center
        common = torch.where(within_box.view(-1, 1), common/width, common)
        common = torch.where(~within_box.view(-1, 1), common * width - .5*(width-1)*(width-1/width), common)
        return torch.linalg.norm(common, dim=-1, ord=self.pnorm)

    def _calc(self, hbase, hbump, r_base_1, r_delta_1, r_base_2, r_delta_2, tbase, tbump):
        box_low_1, box_high_1, box_center_1, box_width_1 = self.get_box(r_base_1, r_delta_1)
        box_low_2, box_high_2, box_center_2, box_width_2 = self.get_box(r_base_2, r_delta_2)

        dist_1 = self.dist(hbase, tbump, box_low_1, box_high_1, box_center_1, box_width_1)
        dist_2 = self.dist(tbase, hbump, box_low_2, box_high_2, box_center_2, box_width_2)

        return -dist_1 - dist_2

    def return_score(self, is_predict=False):
        (head_emb, rel_emb, tail_emb) = self.current_batch

        def to_hyper(x):
            return torch.tanh(x)

        # Transform all embeddings to hyperbolic space.
        hbase, hbump = to_hyper(head_emb["e_base"]), to_hyper(head_emb["e_bump"])
        tbase, tbump = to_hyper(tail_emb["e_base"]), to_hyper(tail_emb["e_bump"])
        r_base_1, r_delta_1 = to_hyper(rel_emb["r_base_1"]), to_hyper(rel_emb["r_delta_1"])
        r_base_2, r_delta_2 = to_hyper(rel_emb["r_base_2"]), to_hyper(rel_emb["r_delta_2"])

        return self._calc(hbase, hbump, r_base_1, r_delta_1, r_base_2, r_delta_2, tbase, tbump)
