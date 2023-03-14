import torch
from Models.Model import Model


class TransSparseDT(Model):
    """
    Liang Chang, Manli Zhu, Tianlong Gu, Chenzhong Bin, Junyan Qian, Ji Zhang: Knowledge Graph Embedding by Dynamic
        Translation. IEEE Access 5: 20898-20907 (2017).
    """
    def __init__(self, ent_total, rel_total, dim_e, dim_r, pred_count, pred_loc_count, norm=2,
                 variant="share", sp_deg_min=.0, z=.3):
        """
            dim_e (int): Number of dimensions for entity embeddings
            dim_r (int): Number of dimensions for relation embeddings
            norm (int): L1 or L2 norm. Default: 2 (could be 1).
            variant can be either share or separate
            pred_count is a dictionary pred_count[r]['global']=x such that r is a relation and x is how many triples
                            has r as relation (in the current split).
            pred_loc_count is a dictionary pred_loc_count[r]['domain']=x such that r is a relation and x is how many
                            entities are head for relation r (in the current split). Also, pred_loc_count[r]['range']=y,
                            y is how many entities are tail for relation r (in the current split).
            sp_deg_min (float): Minimum sparse degree. Default: 0.
            z (float): The L2-norm of the alpha parameters. From the paper: "...the L2-norm of them are values from the
                set {.1, .2, .3}."
        """
        super(TransSparseDT, self).__init__(ent_total, rel_total)
        self.dim_e = dim_e
        self.dim_r = dim_r
        self.pnorm = norm
        self.variant = variant
        self.z = z

        if variant == 'share':
            pc = pred_count
            locations = ['global']
        elif variant == 'separate':
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

    def get_score_sign(self):
        # It is a distance (norm).
        return -1

    def initialize_model(self):
        self.create_embedding(self.dim_e, emb_type="entity", name="e")
        self.create_embedding(self.dim_r, emb_type="relation", name="r")

        self.create_embedding(self.dim_r, emb_type="entity", name="ehalpha")
        self.create_embedding(self.dim_r, emb_type="entity", name="etalpha")
        self.create_embedding(self.dim_r, emb_type="relation", name="ralpha")

        if self.variant == 'share':
            names_locations = [('m', 'global')]
        elif self.variant == 'separate':
            names_locations = [('mh', 'domain'), ('mt', 'range')]

        for (name, loc) in names_locations:
            self.create_embedding((self.dim_r, self.dim_e), emb_type="relation", name=name)

            def make_sparse(matrix, deg):
                with torch.no_grad():
                    torch.nn.functional.dropout(matrix, p=deg, inplace=True)

            e = self.get_embedding(emb_type="relation", name=name)
            for r in self.sparse_degrees:
                make_sparse(e.emb[r], self.sparse_degrees[r][loc])

        self.register_scale_constraint(emb_type="entity", name="e")
        self.register_scale_constraint(emb_type="relation", name="r")

        # Scale constraints of the alpha parameters.
        self.register_scale_constraint(emb_type="entity", name="ehalpha", z=self.z)
        self.register_scale_constraint(emb_type="entity", name="etalpha", z=self.z)
        self.register_scale_constraint(emb_type="relation", name="ralpha", z=self.z)

    def get_et(self, m, e):
        batch_size = e.shape[0]
        return torch.matmul(m, e.view(batch_size, -1, 1)).view(batch_size, self.dim_r)

    def _calc(self, h, mh, halpha, r, ralpha, t, mt, talpha, is_predict):
        ht = self.get_et(mh, h)
        tt = self.get_et(mt, t)
        if not is_predict:
            self.onthefly_constraints.append(self.scale_constraint(ht))
            self.onthefly_constraints.append(self.scale_constraint(tt))
        return torch.pow(torch.linalg.norm((ht + halpha) + (r + ralpha) - (tt + talpha), dim=-1, ord=self.pnorm), 2)

    def return_score(self, is_predict=False):
        (head_emb, rel_emb, tail_emb) = self.current_batch

        h, halpha = head_emb["e"], head_emb["ehalpha"]
        t, talpha = tail_emb["e"], tail_emb["etalpha"]
        r, ralpha = rel_emb["r"], rel_emb["ralpha"]

        # When share, mh and mt are the same.
        if self.variant == 'share':
            mh, mt = rel_emb["m"], rel_emb["m"]
        elif self.variant == 'separate':
            mh, mt = rel_emb["mh"], rel_emb["mt"]

        return self._calc(h, mh, halpha, r, ralpha, t, mt, talpha, is_predict)
