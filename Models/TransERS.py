import torch
from Models.Model import Model


class TransERS(Model):
    def __init__(self, ent_total, rel_total, dim, norm=2):
        super(TransERS, self).__init__(ent_total, rel_total)
        self.dim = dim
        self.pnorm = norm

    def get_default_loss(self):
        return 'margin'

    def initialize_model(self):
        self.create_embedding(self.dim, emb_type="entity", name="e", norm_method="norm")
        self.create_embedding(self.dim, emb_type="relation", name="r")
        self.register_custom_extra_loss(self.limit_based_loss)

    def limit_based_loss(self, head_emb, rel_emb, tail_emb, pos_neg, margin=1e-4, lmbda=.5):
        loss = torch.nn.MarginRankingLoss(-margin)

        # Get only positives.
        h = head_emb["e"][pos_neg == 1]
        t = tail_emb["e"][pos_neg == 1]
        r = rel_emb["r"][pos_neg == 1]
        pos_scores = self._calc(h, r, t)

        targets = torch.ones(len(pos_scores))
        zeros = torch.zeros_like(pos_scores)
        if pos_scores.is_cuda:
            targets = targets.cuda()
            zeros = zeros.cuda()

        return lmbda * loss(pos_scores, zeros, targets)

    def _calc(self, h, r, t):
        return -torch.linalg.norm(h + r - t, dim=-1, ord=self.pnorm)

    def return_score(self, is_predict=False):
        (head_emb, rel_emb, tail_emb) = self.current_batch

        h = head_emb["e"]
        t = tail_emb["e"]
        r = rel_emb["r"]

        return self._calc(h, r, t)
