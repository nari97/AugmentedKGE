import torch
from Models.Model import Model


class GTrans(Model):
    def __init__(self, ent_total, rel_total, dim, head_context, tail_context, norm=2):
        super(GTrans, self).__init__(ent_total, rel_total)
        self.dim = dim
        self.head_context = head_context
        self.tail_context = tail_context
        self.pnorm = norm

    def get_default_loss(self):
        return 'margin'

    def initialize_model(self):
        self.create_embedding(self.dim, emb_type="entity", name="ea")
        self.create_embedding(self.dim, emb_type="entity", name="ee")
        self.create_embedding(self.rel_tot, emb_type="entity", name="bh", init_params=[1e-10, 1])
        self.create_embedding(self.rel_tot, emb_type="entity", name="bt", init_params=[1e-10, 1])

        self.create_embedding(self.dim, emb_type="relation", name="ra")
        self.create_embedding(self.dim, emb_type="relation", name="re")

        # Special init. The original paper considers M_e as well, but the formulation is not correct: M_e(e_i, r_j)
        #   does not involve r_j but e_j.
        bh, bt = self.get_embedding('entity', "bh"), self.get_embedding('entity', "bt")
        for e in range(0, self.ent_tot):
            # When e as head, check head_context and update bh; otherwise, check tail context and update bt.
            for ctx, b in [(self.head_context, bh), (self.tail_context, bt)]:
                total_e = 0
                for r in range(0, self.rel_tot):
                    if e in ctx[r].keys():
                        total_e += len(ctx[r][e])

                for r in range(0, self.rel_tot):
                    if e in ctx[r].keys():
                        b.emb.data[e][r] += len(ctx[r][e])/total_e

        self.register_scale_constraint(emb_type="entity", name="ee")
        self.register_scale_constraint(emb_type="relation", name="re")

    def get_et(self, ee, ea, ra):
        batch_size = ee.shape[0]
        m = torch.matmul(ra.view(batch_size, -1, 1), ea.view(batch_size, 1, -1))
        return torch.matmul(m, ee.view(batch_size, -1, 1)).view(batch_size, self.dim)

    def get_entity(self, ee, ea, ra, b):
        alpha = torch.gather(b, 1, self.current_data['batch_r'].view(-1, 1))
        return (1 - alpha) * ee + alpha * self.get_et(ee, ea, ra)

    def _calc(self, ha, he, bh, ra, re, ta, te, bt, is_predict):
        h = self.get_entity(he, ha, ra, bh)
        r = re
        t = self.get_entity(te, ta, ra, bt)

        result = h + r - t
        wr = 1/torch.std(result, dim=1)

        # Register on-the-fly constraints.
        if not is_predict:
            self.onthefly_constraints.append(self.scale_constraint(h))
            self.onthefly_constraints.append(self.scale_constraint(t))
            # This returns a single value.
            self.onthefly_constraints.append(self.scale_constraint(wr, ctype='ge').view(-1, 1))

        return -torch.pow(torch.linalg.norm(wr.view(-1, 1) * result, ord=self.pnorm, dim=-1), 2)

    def return_score(self, is_predict=False):
        (head_emb, rel_emb, tail_emb) = self.current_batch

        ha, he, bh = head_emb["ea"], head_emb["ee"], head_emb["bh"]
        ta, te, bt = tail_emb["ea"], tail_emb["ee"], head_emb["bt"]
        ra, re = rel_emb["ra"], rel_emb["re"]

        return self._calc(ha, he, bh, ra, re, ta, te, bt, is_predict)
