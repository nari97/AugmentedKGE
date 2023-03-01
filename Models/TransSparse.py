import torch
from Models.Model import Model


class TransSparse(Model):
    """
    Guoliang Ji, Kang Liu, Shizhu He, Jun Zhao: Knowledge Graph Completion with Adaptive Sparse Transfer Matrix. AAAI
        2016: 985-991.
    """
    def __init__(self, ent_total, rel_total, dim_e, dim_r, pred_count, pred_loc_count, norm=2,
                 variant="share", sp_deg_min=.0):
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
        """
        super(TransSparse, self).__init__(ent_total, rel_total)
        self.dim_e = dim_e
        self.dim_r = dim_r
        self.pnorm = norm
        self.variant = variant

        if variant == 'share':
            pc = pred_count
            locations = ['global']
        elif variant == 'separate':
            pc = pred_loc_count
            locations = ['domain', 'range']

        self.sparse_degrees = {}
        for loc in locations:
            # Find maximum.
            max = -1;
            for r in pc:
                if pc[r][loc] > max:
                    max = pc[r][loc]
            # Compute sparsity (see Eqs. (1) and (3)).
            for r in pc:
                if r not in self.sparse_degrees:
                    self.sparse_degrees[r] = {}
                self.sparse_degrees[r][loc] = 1 - ((1 - sp_deg_min) * pc[r][loc] / max)

    def get_default_loss(self):
        # Eq. (6).
        return 'margin'

    def initialize_model(self):
        # These are common to both variants.
        self.create_embedding(self.dim_e, emb_type="entity", name="e")
        self.create_embedding(self.dim_r, emb_type="relation", name="r")
        # Create one or two embeddings.
        if self.variant == 'share':
            names_locations = [('m', 'global')]
        elif self.variant == 'separate':
            names_locations = [('mh', 'domain'), ('mt', 'range')]

        for (name, loc) in names_locations:
            self.create_embedding((self.dim_r, self.dim_e), emb_type="relation", name=name)

            # Make the matrix sparse with the given degree.
            def make_sparse(matrix, deg):
                # This is to avoid issues with the dropout.
                with torch.no_grad():
                    # The matrix should be sparse!
                    torch.nn.functional.dropout(matrix, p=deg, inplace=True)

            # The paper talks about structured and unstructured sparse matrix. We used unstructured only (see Figure 2).
            e = self.get_embedding(emb_type="relation", name=name)
            for r in self.sparse_degrees:
                make_sparse(e.emb[r], self.sparse_degrees[r][loc])

        # See below Eq. (6).
        self.register_scale_constraint(emb_type="entity", name="e")
        self.register_scale_constraint(emb_type="relation", name="r")

    # This method computes each transfer (see Eqs. (2) and (4)).
    def get_et(self, m, e):
        batch_size = e.shape[0]
        #  multiply by vector and put it back to regular shape.
        return torch.matmul(m, e.view(batch_size, -1, 1)).view(batch_size, self.dim_r)

    def _calc(self, h, mh, r, t, mt, is_predict):
        # Transfers.
        ht = self.get_et(mh, h)
        tt = self.get_et(mt, t)
        # Add the constraints of the transfers (see below Eq. (6)).
        if not is_predict:
            self.onthefly_constraints.append(self.scale_constraint(ht))
            self.onthefly_constraints.append(self.scale_constraint(tt))
        # Eq. (5).
        return -torch.pow(torch.linalg.norm(ht + r - tt, dim=-1, ord=self.pnorm), 2)

    def return_score(self, is_predict=False):
        (head_emb, rel_emb, tail_emb) = self.current_batch

        h = head_emb["e"]
        t = tail_emb["e"]
        r = rel_emb["r"]

        # When share, mh and mt are the same.
        if self.variant == 'share':
            mh, mt = rel_emb["m"], rel_emb["m"]
        elif self.variant == 'separate':
            mh, mt = rel_emb["mh"], rel_emb["mt"]

        return self._calc(h, mh, r, t, mt, is_predict)
