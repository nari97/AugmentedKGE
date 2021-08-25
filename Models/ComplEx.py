import torch
from .Model import Model
from Utils.Embedding import Embedding
from Utils.utils import clamp_norm, normalize

class ComplEx(Model):

    def __init__(self, ent_total, rel_total, dims, norm = 2):
        super(ComplEx, self).__init__(ent_total, rel_total)

        self.dims = dims
        self.norm = norm

        self.entities_real = Embedding(self.ent_tot, self.dims)
        self.entities_img = Embedding(self.ent_tot, self.dims)
        self.relations_real = Embedding(self.rel_tot, self.dims)
        self.relations_img = Embedding(self.rel_tot, self.dims)

    def normalize(self):
        pass

    def _calc(self, h_re, h_im, t_re, t_im, r_re, r_im):
        return torch.sum(
            h_re * t_re * r_re
            + h_im * t_im * r_re
            + h_re * t_im * r_im
            - h_im * t_re * r_im,
            -1
        )

    def forward(self, data):

        batch_h = self.get_batch(data, "h")
        batch_r = self.get_batch(data, "r")
        batch_t = self.get_batch(data, "t")

        h_real = self.entities_real.get_embedding(batch_h)
        h_img = self.entities_img.get_embedding(batch_h)

        r_real = self.relations_real.get_embedding(batch_r)
        r_img = self.relations_img.get_embedding(batch_r)

        t_real = self.entities_real.get_embedding(batch_t)
        t_img = self.entities_img.get_embedding(batch_t)

        score = self._calc(h_real, h_img, t_real, t_img, r_real, r_img).flatten()

        return score

    def predict(self, data):
        score = -self.forward(data)

        return score



