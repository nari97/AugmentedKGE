import torch
from Models.Model import Model


class TransSparse(Model):

    def __init__(self, ent_total, rel_total, dim_e, dim_r, pred_count, pred_loc_count, type, norm=2, sp_deg_min=.7):
        super(TransSparse, self).__init__(ent_total, rel_total)
        self.dim_e = dim_e
        self.dim_r = dim_r
        self.type = type
        self.pnorm = norm

        if type is 'share':
            pc = pred_count
            locations = ['global']
        elif type is 'separate':
            pc = pred_loc_count
            locations = ['domain', 'range']

        self.sparse_degrees = {}
        for loc in locations:
            max = -1;
            for r in pc:
                if pc[r][loc] > max:
                    max = pc[r][loc]
            for r in pc:
                if r not in self.sparse_degrees:
                    self.sparse_degrees[r] = {}
                self.sparse_degrees[r][loc] = 1 - ((1 - sp_deg_min) * pc[r][loc] / max)

    def get_default_loss(self):
        return 'margin'

    def initialize_model(self):
        self.create_embedding(self.dim_e, emb_type="entity", name="e")
        self.create_embedding(self.dim_r, emb_type="relation", name="r")

        if self.type is 'share':
            names_locations = [('m', 'global')]
        elif self.type is 'separate':
            names_locations = [('mh', 'domain'), ('mt', 'range')]

        for (name, loc) in names_locations:
            self.create_embedding((self.dim_r, self.dim_e), emb_type="relation", name=name)

            e = self.get_embedding(emb_type="relation", name=name)
            for r in self.sparse_degrees:
                self.make_sparse(e.emb[r], self.sparse_degrees[r][loc])

            self.register_custom_constraint(self.h_constraint)
            self.register_custom_constraint(self.t_constraint)

        self.register_scale_constraint(emb_type="entity", name="e", p=2)
        self.register_scale_constraint(emb_type="relation", name="r", p=2)

    def make_sparse(self, matrix, deg):
        # This is to avoid issues with the dropout.
        with torch.no_grad():
            # The matrix should be sparse!
            torch.nn.functional.dropout(matrix, p=deg, training=True, inplace=True)

    def h_constraint(self, head_emb, rel_emb, tail_emb, epsilon=1e-5):
        h = head_emb["e"]

        if self.type is 'share':
            name = 'm'
        elif self.type is 'separate':
            name = 'mh'

        m = rel_emb[name]
        return torch.linalg.norm(self.get_et(m, h), ord=2) - epsilon

    def t_constraint(self, head_emb, rel_emb, tail_emb, epsilon=1e-5):
        t = tail_emb["e"]

        if self.type is 'share':
            name = 'm'
        elif self.type is 'separate':
            name = 'mt'

        m = rel_emb[name]
        return torch.linalg.norm(self.get_et(m, t), ord=2) - epsilon

    def get_et(self, m, e):
        batch_size = e.shape[0]
        #  multiply by vector and put it back to regular shape.
        return torch.matmul(m, e.view(batch_size, -1, 1)).view(batch_size, self.dim_r)

    def _calc(self, h, mh, r, t, mt):
        return -torch.pow(torch.linalg.norm(
            self.get_et(mh, h) + r - self.get_et(mt, t), ord=self.pnorm, dim=-1), 2)

    def return_score(self, is_predict=False):
        (head_emb, rel_emb, tail_emb) = self.current_batch

        h = head_emb["e"]
        t = tail_emb["e"]
        r = rel_emb["r"]

        if self.type is 'share':
            mh = rel_emb["m"]
            mt = rel_emb["m"]
        elif self.type is 'separate':
            mh = rel_emb["mh"]
            mt = rel_emb["mt"]

        return self._calc(h, mh, r, t, mt)
