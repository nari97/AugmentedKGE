import torch
from Models.Model import Model


class GTrans(Model):
    """
    Zhen Tan, Xiang Zhao, Yang Fang, Weidong Xiao: GTrans: Generic Knowledge Graph Embedding via Multi-State Entities
        and Dynamic Relation Spaces. IEEE Access 6: 8232-8244 (2018).
    """
    def __init__(self, ent_total, rel_total, dim, head_context, tail_context, norm=2, variant='dw'):
        """
            dim (int): Number of dimensions for embeddings
            norm (int): L1 or L2 norm. Default: 2 (see Eq. (14)).
            head_context: Dictionary with relations as keys. For each relation, the tail is a key and the value is a
                list of heads for that combination of relation and tail.
            tail_context: Dictionary with relations as keys. For each relation, the head is a key and the value is a
                list of tails for that combination of relation and head.
            variant can be either dw (dynamic weights) or sw (static weights). For sw, we use alpha=.5 and beta=.5.
        """
        super(GTrans, self).__init__(ent_total, rel_total)
        self.dim = dim
        self.head_context = head_context
        self.tail_context = tail_context
        self.pnorm = norm
        self.variant = variant
        self.default_alpha = .5

        if self.variant == 'dw':
            # Compute Mr (Eq. (9)) and Me (Eq. (10)) matrices. Note that there is a typo in Eq. (10), Me(ei, ej).
            mr = torch.empty((ent_total, rel_total), dtype=torch.float64, requires_grad=False)
            # For each relation, get each tail and add the total number of heads. Similar for tail_context.
            for r in head_context.keys():
                for t in head_context[r].keys():
                    mr[t, r] += len(head_context[r][t])
            for r in tail_context.keys():
                for h in tail_context[r].keys():
                    mr[h, r] += len(tail_context[r][h])
            # Normalize rows.
            row_sum = torch.sum(mr, dim=1)
            # Avoiding division by zero.
            row_sum = torch.where(row_sum != 0, row_sum, 1.0)
            mr /= row_sum.view(-1, 1)

            # The paper does not specify, but it seems Me is indeed symmetric, that is, direction is not considered.
            me = torch.empty((ent_total, ent_total), dtype=torch.float64, requires_grad=False)
            # For each relation, get each tail and head and add one. Similar for tail_context.
            for r in head_context.keys():
                for t in head_context[r].keys():
                    for h in head_context[r][t]:
                        me[h, t] += 1
                        me[t, h] += 1
            for r in tail_context.keys():
                for h in tail_context[r].keys():
                    for t in tail_context[r][h]:
                        me[h, t] += 1
                        me[t, h] += 1
            # Normalize rows.
            row_sum = torch.sum(me, dim=1)
            # Avoiding division by zero.
            row_sum = torch.where(row_sum != 0, row_sum, 1.0)
            me /= row_sum.view(-1, 1)

            # Eq. (10) except bias.
            self.w = torch.matmul(me+1, mr)

    def get_default_loss(self):
        # Eq. (19).
        return 'margin'

    def get_score_sign(self):
        # It is a distance (norm).
        return -1

    def initialize_model(self):
        # Abstract (a) and eigenstate embeddings.
        self.create_embedding(self.dim, emb_type="entity", name="ea")
        self.create_embedding(self.dim, emb_type="entity", name="ee")
        self.create_embedding(self.dim, emb_type="relation", name="ra")
        self.create_embedding(self.dim, emb_type="relation", name="re")

        if self.variant == 'dw':
            # Bias is for the whole W_alpha matrix, we use it as a global parameter.
            self.create_embedding((self.ent_tot, self.rel_tot), emb_type="global", name="b")

        # Eqs. (20) and (21).
        self.register_scale_constraint(emb_type="entity", name="ee")
        self.register_scale_constraint(emb_type="relation", name="re")

    @staticmethod
    def get_et(ee, ea, ra):
        # Eqs. (4) and (5).
        batch_size = ee.shape[0]
        m = torch.bmm(ra.view(batch_size, -1, 1), ea.view(batch_size, 1, -1))
        return torch.bmm(m, ee.view(batch_size, -1, 1)).view(batch_size, -1)

    def get_entity(self, ee, ea, ra, a, b):
        if self.variant == 'dw':
            # alpha + bias must be between zero and one.
            alpha = (a + b).view(-1, 1)
            min, max = torch.min(alpha, dim=0), torch.max(alpha, dim=0)
            alpha -= min[0]
            alpha /= (max[0] - min[0])
        if self.variant == 'sw':
            alpha = self.default_alpha

        # Eqs. (6), (7), (12) and (13).
        return (1 - alpha) * ee + alpha * GTrans.get_et(ee, ea, ra)

    def _calc(self, ha, he, alpha_h, bh, ra, re, ta, te, alpha_t, bt, is_predict):
        h = self.get_entity(he, ha, ra, alpha_h, bh)
        t = self.get_entity(te, ta, ra, alpha_t, bt)

        result = h + re - t
        wr = 1/torch.std(result, dim=1)

        # Register on-the-fly constraints.
        if not is_predict:
            # Eq. (22).
            self.onthefly_constraints.append(self.scale_constraint(h))
            self.onthefly_constraints.append(self.scale_constraint(t))
            # Eq. (23). This returns a single value.
            self.onthefly_constraints.append(self.scale_constraint(wr, ctype='ge').view(1))

        return torch.pow(torch.linalg.norm(wr.view(-1, 1) * result, ord=self.pnorm, dim=-1), 2)

    def return_score(self, is_predict=False):
        (head_emb, rel_emb, tail_emb) = self.current_batch

        alpha_h, alpha_t, bh, bt = None, None, None, None
        if self.variant == 'dw':
            alpha_h, alpha_t = self.w[self.current_data["batch_h"], self.current_data["batch_r"]], \
                self.w[self.current_data["batch_t"], self.current_data["batch_r"]]
            bh, bt = self.current_global_embeddings["b"][0, self.current_data["batch_h"], self.current_data["batch_r"]], \
                self.current_global_embeddings["b"][0, self.current_data["batch_t"], self.current_data["batch_r"]]

        ha, he = head_emb["ea"], head_emb["ee"]
        ta, te = tail_emb["ea"], tail_emb["ee"]
        ra, re = rel_emb["ra"], rel_emb["re"]

        return self._calc(ha, he, alpha_h, bh, ra, re, ta, te, alpha_t, bt, is_predict)
