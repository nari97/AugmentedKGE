import torch
from .Model import Model


class ComplEx(Model):
    def __init__(self, ent_total, rel_total, dim):
        super(ComplEx, self).__init__(ent_total, rel_total)
        self.dim = dim

    def get_default_loss(self):
        # http://proceedings.mlr.press/v48/trouillon16-supp.pdf
        return 'soft'

    def initialize_model(self):
        self.create_embedding(self.dim, emb_type="entity", name="e_real")
        self.create_embedding(self.dim, emb_type="entity", name="e_img")
        self.create_embedding(self.dim, emb_type="relation", name="r_real")
        self.create_embedding(self.dim, emb_type="relation", name="r_img")

        # http://proceedings.mlr.press/v48/trouillon16-supp.pdf
        self.register_complex_regularization(emb_type="entity", name_real="e_real", name_img="e_img")
        self.register_complex_regularization(emb_type="relation", name_real="r_real", name_img="r_img")
        
    def _calc(self, h_re, h_im, t_re, t_im, r_re, r_im):
        return torch.sum(h_re * t_re * r_re + h_im * t_im * r_re +
                         h_re * t_im * r_im - h_im * t_re * r_im, -1)

    def return_score(self, is_predict=False):
        (head_emb, rel_emb, tail_emb) = self.current_batch

        h_real, h_img = head_emb["e_real"], head_emb["e_img"]
        t_real, t_img = tail_emb["e_real"], tail_emb["e_img"]
        r_real, r_img = rel_emb["r_real"], rel_emb["r_img"]

        return self._calc(h_real, h_img, t_real, t_img, r_real, r_img).flatten()
