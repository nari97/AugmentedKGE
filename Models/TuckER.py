import torch
from Models.Model import Model


class TuckER(Model):

    def __init__(self, ent_total, rel_total, dim_e, dim_r):
        """
        Args:
            ent_total (int): Total number of entities
            rel_total (int): Total number of relations
            dim_e (int): Number of dimensions for entity embeddings
            dim_r (int): Number of dimensions for relation embeddings
        """
        super(TuckER, self).__init__(ent_total, rel_total)
        self.dim_e = dim_e
        self.dim_r = dim_r

    def initialize_model(self):
        self.create_embedding(self.dim_e, emb_type="entity", name="e")
        self.create_embedding(self.dim_r, emb_type="relation", name="r")
        self.create_embedding((self.dim_e, self.dim_r, self.dim_e), emb_type="global", name="w")

    def _calc(self, h, r, t, w, is_predict=False):
        batch_size = h.shape[0]

        # This changes from (batch_size, dim_e/dim_r) to (batch_size, 1, dim_e/dim_r)
        bh = h.view(batch_size, 1, -1)
        br = r.view(batch_size, 1, -1)
        bt = t.view(batch_size, 1, -1)

        # First, W x_1 h; roll W: (dime_e, dim_r*dim_e).
        w_rolled = w.view(self.dim_e, -1)
        # Multiply: (batch_size, 1, dim_e) x (dime_e, dim_r*dim_e) = (batch_size, 1, dim_r*dim_e)
        prod_one = torch.matmul(bh, w_rolled)
        # Unroll from (batch_size, 1, dim_r*dim_e) to (batch_size, 1, dim_r, dim_e)
        prod_one = prod_one.view(batch_size, 1, self.dim_r, self.dim_e)

        # Second, (W x_1 h) x_2 r; roll prod_one: (batch_size, dim_r, dim_e)
        prod_one_rolled = prod_one.view(batch_size, self.dim_r, -1)
        # Multiply: (batch_size, 1, dim_r) x (batch_size, dim_r, dim_e) = (batch_size, 1, dim_e)
        prod_two = torch.bmm(br, prod_one_rolled)
        # Unroll from (batch_size, 1, dim_e) to (batch_size, 1, 1, dim_e)
        prod_two = prod_two.view(batch_size, -1, 1, self.dim_e)

        # Third, ((W x_1 h) x_2 r) x_3 t; roll prod_two: (batch_size, dim_e, 1)
        prod_two_rolled = prod_two.view(batch_size, self.dim_e, -1)
        # Multiply: (batch_size, 1, dim_e) x (batch_size, dim_e, 1) = (batch_size, 1, 1)
        scores = torch.bmm(bt, prod_two_rolled)

        # Apply only sigmoid when predicting.
        if is_predict:
            scores = torch.sigmoid(scores)

        return scores

    def return_score(self, head_emb, rel_emb, tail_emb, is_predict=False):
        h = head_emb["e"]
        t = tail_emb["e"]
        r = rel_emb["r"]
        w = self.get_embedding("global", "w").emb
        w = w.to(h.device)

        return self._calc(h, r, t, w, is_predict)
