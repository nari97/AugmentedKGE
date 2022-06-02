import math
import torch
from Models.Model import Model


class TransM(Model):

    def __init__(self, ent_total, rel_total, dim, pred_count, pred_loc_count, norm=2):
        super(TransM, self).__init__(ent_total, rel_total)
        self.dim = dim
        self.pnorm = norm

        self.w_r = {}
        for r in pred_count.keys():
            self.w_r[r] = 1/math.log(pred_count[r]['global']/pred_loc_count[r]['domain'] +
                                     pred_count[r]['global']/pred_loc_count[r]['range'])

    def get_default_loss(self):
        return 'margin'

    def initialize_model(self):
        self.create_embedding(self.dim, emb_type="entity", name="e", norm_method="norm")
        self.create_embedding(self.dim, emb_type="relation", name="r")
        wr = self.create_embedding(1, emb_type="relation", name="wr")
        # We model w_r as an embedding that we are not going to update during autograd.
        for r in self.w_r.keys():
            wr.emb.data[r] = self.w_r[r]

    @torch.utils.hooks.unserializable_hook
    def no_derivative(self, grad):
        return torch.zeros_like(grad)

    def _calc(self, h, r, wr, t):
        return -wr.flatten()*torch.pow(torch.linalg.norm(h + r - t, dim=-1, ord=self.pnorm), 2)

    def return_score(self, is_predict=False):
        (head_emb, rel_emb, tail_emb) = self.current_batch

        h = head_emb["e"]
        t = tail_emb["e"]
        r, wr = rel_emb["r"], rel_emb["wr"]

        # Only during training.
        if not is_predict:
            wr.register_hook(self.no_derivative)

        return self._calc(h, r, wr, t)
