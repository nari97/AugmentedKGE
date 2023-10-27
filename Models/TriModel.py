import torch
from Models.Model import Model


class TriModel(Model):
    """
    Sameh K. Mohamed, Vít Novácek: Link Prediction Using Multi Part Embeddings. ESWC 2019: 240-254.
    """
    def __init__(self, ent_total, rel_total, dim):
        super(TriModel, self).__init__(ent_total, rel_total)
        self.dim = dim

    def get_default_loss(self):
        # Eq. (11).
        return 'soft'

    def get_score_sign(self):
        # The paper uses soft loss, so that means positive scores will be larger than negative scores.
        return 1

    def initialize_model(self):
        # Eq. (10). Embedding initialization mentioned in Section 4.3.
        # Regularization is mentioned in Eq. (11).
        self.create_embedding(self.dim, emb_type="entity", name="e_1", reg=True)
        self.create_embedding(self.dim, emb_type="entity", name="e_2", reg=True)
        self.create_embedding(self.dim, emb_type="entity", name="e_3", reg=True)
        self.create_embedding(self.dim, emb_type="relation", name="r_1")
        self.create_embedding(self.dim, emb_type="relation", name="r_2")
        self.create_embedding(self.dim, emb_type="relation", name="r_3")
        
    def _calc(self, h, r, t):
        (h_1, h_2, h_3), (r_1, r_2, r_3), (t_1, t_2, t_3) = h, r, t
        # Eq. (10).
        return torch.sum(h_1 * r_1 * t_3 + h_2 * r_2 * t_2 + h_3 * r_3 * t_3, -1)

    def return_score(self, is_predict=False):
        (head_emb, rel_emb, tail_emb) = self.current_batch

        h_1, h_2, h_3 = head_emb["e_1"], head_emb["e_2"], head_emb["e_3"]
        t_1, t_2, t_3 = tail_emb["e_1"], tail_emb["e_2"], tail_emb["e_3"]
        r_1, r_2, r_3 = rel_emb["r_1"], rel_emb["r_2"], rel_emb["r_3"]

        return self._calc((h_1, h_2, h_3), (r_1, r_2, r_3), (t_1, t_2, t_3)).flatten()
