import torch
from Models.Model import Model


class TransDR(Model):
    """
    Zhen Tan, Xiang Zhao, Yang Fang, Weidong Xiao, Jiuyang Tang: Knowledge Representation Learning via Dynamic Relation
        Spaces. ICDM Workshops 2016: 684-691.
    """
    def __init__(self, ent_total, rel_total, dim, norm=2):
        """
            dim (int): Number of dimensions for embeddings. In Section 3.B the paper mentions a single dimension m; in
                        the experiments, two dimensions m and n are mentioned. We use a single dimension.
            norm (int): L1 or L2 norm. Default: 2
        """
        super(TransDR, self).__init__(ent_total, rel_total)
        self.dim = dim
        self.pnorm = norm

    def get_default_loss(self):
        return 'margin'

    def get_score_sign(self):
        # It is a distance (norm).
        return -1

    def initialize_model(self):
        # Section 3.B.
        self.create_embedding(self.dim, emb_type="entity", name="ee")
        self.create_embedding(self.dim, emb_type="entity", name="em")
        self.create_embedding(self.dim, emb_type="relation", name="re")
        self.create_embedding(self.dim, emb_type="relation", name="rm")
        # Eqs. (5) and (6).
        self.register_scale_constraint(emb_type="entity", name="ee")
        self.register_scale_constraint(emb_type="relation", name="re")

    def get_et(self, e, em, rm):
        # Eqs. (3) and (4).
        batch_size = em.shape[0]

        # ep changed into a row matrix, and rp changed into a column matrix.
        m = torch.matmul(rm.view(batch_size, -1, 1), em.view(batch_size, 1, -1))

        # multiply by vector and put it back to regular shape.
        return torch.matmul(m, e.view(batch_size, -1, 1)).view(batch_size, self.dim) + e

    def _calc(self, he, hm, re, rm, te, tm, is_predict):
        # Transfers.
        h = self.get_et(he, hm, rm)
        t = self.get_et(te, tm, rm)
        # Score function (after Eq. (4)); assuming r is re.
        result = h + re - t
        # See above training paragraph.
        wr = 1 / torch.pow(torch.std(result, dim=-1), 2)
        # Apply constraints only during training.
        if not is_predict:
            # See Eq. (8).
            self.onthefly_constraints.append(self.scale_constraint(h))
            self.onthefly_constraints.append(self.scale_constraint(t))
            # The paper says ||wr||=1, but they implement it as ||wr||>=1 (see Eq. 8).
            self.onthefly_constraints.append(self.scale_constraint(wr, ctype='ge').view(1))
        return torch.pow(torch.linalg.norm(wr.view(-1, 1) * result, dim=-1, ord=self.pnorm), 2)

    def return_score(self, is_predict=False):
        (head_emb, rel_emb, tail_emb) = self.current_batch

        he, hm = head_emb["ee"], head_emb["em"]
        te, tm = tail_emb["ee"], tail_emb["em"]
        re, rm = rel_emb["re"], rel_emb["rm"]

        return self._calc(he, hm, re, rm, te, tm, is_predict)
