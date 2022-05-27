import torch
from .Model import Model


class ComplEx(Model):
    def __init__(self, ent_total, rel_total, dim):
        super(ComplEx, self).__init__(ent_total, rel_total)
        self.dim = dim

    def get_default_loss(self):
        return 'bce'

    def initialize_model(self):
        self.create_embedding(self.dim, emb_type="entity", name="e_real")
        self.create_embedding(self.dim, emb_type="relation", name="r_real")
        self.create_embedding(self.dim, emb_type="entity", name="e_img")
        self.create_embedding(self.dim, emb_type="relation", name="r_img")

        self.register_scale_constraint(emb_type="entity", name="e_real", p=2)
        self.register_scale_constraint(emb_type="entity", name="e_img", p=2)
        self.register_scale_constraint(emb_type="relation", name="r_real", p=2)
        self.register_scale_constraint(emb_type="relation", name="r_img", p=2)
        
    def _calc(self, h_re, h_im, t_re, t_im, r_re, r_im):
        return torch.sum(h_re * t_re * r_re + h_im * t_im * r_re +
                         h_re * t_im * r_im - h_im * t_re * r_im, -1)

    def return_score(self, is_predict=False):
        (head_emb, rel_emb, tail_emb) = self.current_batch

        h_real = head_emb["e_real"]
        h_img = head_emb["e_img"]
        t_real = tail_emb["e_real"]
        t_img = tail_emb["e_img"]

        r_real = rel_emb["r_real"]
        r_img = rel_emb["r_img"]

        return self._calc(h_real, h_img, t_real, t_img, r_real, r_img).flatten()
