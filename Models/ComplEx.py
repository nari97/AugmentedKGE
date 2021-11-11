import torch
from .Model import Model



class ComplEx(Model):

    def __init__(self, ent_total, rel_total, dims, norm = 2, inner_norm = False):
        super(ComplEx, self).__init__(ent_total, rel_total, dims, "complex", inner_norm)

        self.create_embedding(self.dims, emb_type = "entity", name = "e_real", normMethod = "clamp", norm_params = self.norm_params)
        self.create_embedding(self.dims, emb_type = "relation", name = "r_real", normMethod = "clamp", norm_params= self.norm_params)
        self.create_embedding(self.dims, emb_type = "entity", name = "e_img", normMethod = "clamp", norm_params = self.norm_params)
        self.create_embedding(self.dims, emb_type = "relation", name = "r_img", normMethod = "clamp", norm_params= self.norm_params)

        self.register_params()
        
    def _calc(self, h_re, h_im, t_re, t_im, r_re, r_im):
        return torch.sum(
            h_re * t_re * r_re
            + h_im * t_im * r_re
            + h_re * t_im * r_im
            - h_im * t_re * r_im,
            -1
        )

    def returnScore(self, head_emb, rel_emb, tail_emb):

        h_real = head_emb["e_real"]
        h_img = head_emb["e_img"]

        r_real = rel_emb["r_real"]
        r_img = rel_emb["r_img"]

        t_real = tail_emb["e_real"]
        t_img = tail_emb["e_img"]


        score = self._calc(h_real, h_img, t_real, t_img, r_real, r_img).flatten()

        return score

    



