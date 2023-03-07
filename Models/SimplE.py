import torch
from Models.Model import Model


class SimplE(Model):
    """
    Seyed Mehran Kazemi, David Poole: SimplE Embedding for Link Prediction in Knowledge Graphs. NeurIPS 2018: 4289-4300.
    """
    def __init__(self, ent_total, rel_total, dim, variant='both'):
        """
            dim (int): Number of dimensions for embeddings
            variant can be either both or ignr (ignore inverse)
        """
        super(SimplE, self).__init__(ent_total, rel_total)
        self.dim = dim
        self.variant = variant

    def get_default_loss(self):
        # The "Learning SimplE Models" paragraph mentions soft.
        return 'soft'

    def get_score_sign(self):
        # It is a similarity function.
        return 1

    def initialize_model(self):
        # The "Learning SimplE Models" paragraph mentions L2 regularization.
        self.create_embedding(self.dim, emb_type="entity", name="he", reg=True)
        self.create_embedding(self.dim, emb_type="entity", name="te", reg=True)
        self.create_embedding(self.dim, emb_type="relation", name="r", reg=True)
        self.create_embedding(self.dim, emb_type="relation", name="r_inv", reg=True)

    def _calc_avg(self, hei, hej, tei, tej, r, r_inv):
        # See Section 4. The <.> operation is defined in Section 2.
        return (torch.sum(hei * r * tej, -1) + torch.sum(hej * r_inv * tei, -1))/2

    def _calc_ignr(self, h, r, t):
        # inv is ignored after training.
        return torch.sum(h * r * t, -1)

    def return_score(self, is_predict=False):
        (head_emb, rel_emb, tail_emb) = self.current_batch

        hei, hej = head_emb["he"], tail_emb["he"]
        tei, tej = head_emb["te"], tail_emb["te"]
        r, r_inv = rel_emb["r"], rel_emb["r_inv"]

        if self.variant == 'ignr' and is_predict:
            return self._calc_ignr(hei, r, tej)
        else:
            return self._calc_avg(hei, hej, tei, tej, r, r_inv)
