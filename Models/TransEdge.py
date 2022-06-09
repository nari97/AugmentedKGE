import torch
from Models.Model import Model


class TransEdge(Model):
    def __init__(self, ent_total, rel_total, dim, norm=2):
        super(TransEdge, self).__init__(ent_total, rel_total)
        self.dim = dim
        self.pnorm = norm
        self.activation = torch.nn.Tanh()

    def get_default_loss(self):
        return 'limit'

    def initialize_model(self):
        self.create_embedding(self.dim, emb_type="entity", name="e")
        self.create_embedding(self.dim, emb_type="entity", name="ec")
        self.create_embedding(self.dim, emb_type="relation", name="r")

    def _calc(self, h, hc, r, t, tc, is_predict):
        # https://github.com/nju-websoft/TransEdge/blob/master/code/context_operator.py#L5
        htc = torch.concat([hc, tc], 1)
        # Two layers.
        for i in range(2):
            htc = torch.linalg.norm(htc, dim=1, ord=self.pnorm, keepdim=True)
            layer = torch.nn.Linear(1, self.dim, bias=True, dtype=h.dtype)
            htc = layer(htc)
            htc = self.activation(htc)
        htc = torch.linalg.norm(htc, dim=1, ord=self.pnorm, keepdim=True)

        if not is_predict:
            # The paper says ||htc||=1; norm not specified (assuming 2); we implement ||htc||_2<=1 and ||htc||_2>=1.
            self.onthefly_constraints.append(self.scale_constraint(htc))
            self.onthefly_constraints.append(self.scale_constraint(htc, ctype='ge'))

        psi = r - torch.sum(r * htc, dim=1, keepdim=True) * htc

        return -torch.linalg.norm(h + psi - t, dim=-1, ord=self.pnorm)

    def return_score(self, is_predict=False):
        (head_emb, rel_emb, tail_emb) = self.current_batch

        h = head_emb["e"]
        hc = head_emb["ec"]
        t = tail_emb["e"]
        tc = tail_emb["ec"]
        r = rel_emb["r"]

        return self._calc(h, hc, r, t, tc, is_predict)
