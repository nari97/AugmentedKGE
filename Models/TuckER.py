import torch
from Models.Model import Model


class TuckER(Model):
    """
    Ivana Balazevic, Carl Allen, Timothy M. Hospedales: TuckER: Tensor Factorization for Knowledge Graph Completion.
        EMNLP/IJCNLP (1) 2019: 5184-5193.
    The authors used batch normalization and dropout during training. We are not using any of those.
    """
    def __init__(self, ent_total, rel_total, dim_e, dim_r, apply_sigmoid=False):
        """
        Args:
            ent_total (int): Total number of entities
            rel_total (int): Total number of relations
            dim_e (int): Number of dimensions for entity embeddings
            dim_r (int): Number of dimensions for relation embeddings
            apply_sigmoid (Bool): Whether sigmoid must be applied to scores during training. Note that BCEWithLogitsLoss
                already applies sigmoid, so, if this is the loss function used, apply_sigmoid must be set to False. If
                a different loss function is applied, then apply_sigmoid should be set to True.
        """
        super(TuckER, self).__init__(ent_total, rel_total)
        self.dim_e = dim_e
        self.dim_r = dim_r
        self.apply_sigmoid = apply_sigmoid

    def get_default_loss(self):
        # Eq. (3).
        return 'bce'

    def initialize_model(self):
        # See below Eq. (2).
        self.create_embedding(self.dim_e, emb_type="entity", name="e")
        self.create_embedding(self.dim_r, emb_type="relation", name="r")
        # This is the core tensor.
        self.create_embedding((self.dim_e, self.dim_r, self.dim_e), emb_type="global", name="w")

    def _calc(self, h, r, t, w, is_predict):
        # This is the implementation of Eq. (2). So easy!
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

        # Apply only sigmoid when predicting. From the paper: "...We apply logistic sigmoid to each score..."
        # Apply sigmoid when predicting or when indicated by apply_sigmoid.
        if is_predict or self.apply_sigmoid:
            scores = torch.sigmoid(scores)

        return scores

    def return_score(self, is_predict=False):
        (head_emb, rel_emb, tail_emb) = self.current_batch

        h = head_emb["e"]
        t = tail_emb["e"]
        r = rel_emb["r"]
        w = self.current_global_embeddings["w"]

        return self._calc(h, r, t, w, is_predict)
