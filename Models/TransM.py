import math
import torch
from Models.Model import Model


class TransM(Model):
    """
    Miao Fan, Qiang Zhou, Emily Chang, Thomas Fang Zheng: Transition-based Knowledge Graph Embedding with Relational
        Mapping Properties. PACLIC 2014: 328-337.
    """
    def __init__(self, ent_total, rel_total, dim, pred_count, pred_loc_count, norm=2):
        """
            dim (int): Number of dimensions for embeddings
            pred_count is a dictionary pred_count[r]['global']=x such that r is a relation and x is how many triples
                            has r as relation (in the current split).
            pred_loc_count is a dictionary pred_loc_count[r]['domain']=x such that r is a relation and x is how many
                            entities are head for relation r (in the current split). Also, pred_loc_count[r]['range']=y,
                            y is how many entities are tail for relation r (in the current split).
            norm (int): L1 or L2 norm. Default: 2. From the paper: "...we use the inner product..."
        """
        super(TransM, self).__init__(ent_total, rel_total)
        self.dim = dim
        self.pnorm = norm
        # Computing w_r values for each relation. See Section 3.2.
        self.w_r = {}
        for r in pred_count.keys():
            self.w_r[r] = 1/math.log(pred_count[r]['global']/pred_loc_count[r]['domain'] +
                                     pred_count[r]['global']/pred_loc_count[r]['range'])

    def get_default_loss(self):
        # Eq. (2).
        return 'margin'

    def initialize_model(self):
        self.create_embedding(self.dim, emb_type="entity", name="e", norm_method="norm")
        self.create_embedding(self.dim, emb_type="relation", name="r")
        wr = self.create_embedding(1, emb_type="relation", name="wr")
        # We model w_r as an embedding that we are not going to update during autograd.
        for r in self.w_r.keys():
            wr.emb.data[r] = self.w_r[r]
        wr.requires_grad_(False)

    def _calc(self, h, r, wr, t):
        # Eq. (1). The power of 2 is mentioned in Section 3.3.
        return -wr.flatten() * torch.pow(torch.linalg.norm(h + r - t, dim=-1, ord=self.pnorm), 2)

    def return_score(self, is_predict=False):
        (head_emb, rel_emb, tail_emb) = self.current_batch

        h = head_emb["e"]
        t = tail_emb["e"]
        r, wr = rel_emb["r"], rel_emb["wr"]

        return self._calc(h, r, wr, t)
