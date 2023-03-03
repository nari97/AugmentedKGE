import torch
from Models.Model import Model


class ComplEx(Model):
    """
    Théo Trouillon, Johannes Welbl, Sebastian Riedel, Éric Gaussier, Guillaume Bouchard:
        Complex Embeddings for Simple Link Prediction. ICML 2016: 2071-2080.
    Supplementary: http://proceedings.mlr.press/v48/trouillon16-supp.pdf
    """
    def __init__(self, ent_total, rel_total, dim):
        super(ComplEx, self).__init__(ent_total, rel_total)
        self.dim = dim

    def get_default_loss(self):
        # Eq. (12).
        return 'soft'

    def get_score_sign(self):
        # The paper uses soft loss, so that means positive scores will be larger than negative scores.
        return 1

    def initialize_model(self):
        # Eq. (11)
        # Regularization is mentioned in Eq. (12). In the supplement, it is mentioned that it is equivalent to
        #   L2 regularization over real and image, independently.
        self.create_embedding(self.dim, emb_type="entity", name="e_real", reg=True)
        self.create_embedding(self.dim, emb_type="entity", name="e_img", reg=True)
        self.create_embedding(self.dim, emb_type="relation", name="r_real", reg=True)
        self.create_embedding(self.dim, emb_type="relation", name="r_img", reg=True)
        
    def _calc(self, h_re, h_im, t_re, t_im, r_re, r_im):
        # Eq. (11).
        return torch.sum(h_re * t_re * r_re + h_im * t_im * r_re +
                         h_re * t_im * r_im - h_im * t_re * r_im, -1)

    def return_score(self, is_predict=False):
        (head_emb, rel_emb, tail_emb) = self.current_batch

        h_real, h_img = head_emb["e_real"], head_emb["e_img"]
        t_real, t_img = tail_emb["e_real"], tail_emb["e_img"]
        r_real, r_img = rel_emb["r_real"], rel_emb["r_img"]

        return self._calc(h_real, h_img, t_real, t_img, r_real, r_img).flatten()
