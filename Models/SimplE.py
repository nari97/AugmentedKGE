import torch
from .Model import Model


class SimplE(Model):

    def __init__(self, ent_total, rel_total, dims):
        super(SimplE, self).__init__(ent_total, rel_total, dims, "simple")

        self.create_embedding(self.dims, emb_type = "entity", name = "h", normMethod = "none", norm_params = self.norm_params)
        self.create_embedding(self.dims, emb_type = "relation", name = "r", normMethod = "none", norm_params= self.norm_params)
        self.create_embedding(self.dims, emb_type = "entity", name = "t", normMethod = "none", norm_params = self.norm_params)
        self.create_embedding(self.dims, emb_type = "relation", name = "r_inv", normMethod = "none", norm_params= self.norm_params)

        self.register_params()
        

    def _calc_avg(self, h_i, t_i, h_j, t_j, r, r_inv):
        return (torch.sum(h_i * r * t_j, -1) + torch.sum(h_j * r_inv * t_i, -1))/2

    def _calc_ingr(self, h, r, t):
        return torch.sum(h * r * t, -1)

    def forward(self, data):

        h_i = self.embeddings["entity"]["h"].get_embedding(data["batch_h"])
        h_j = self.embeddings["entity"]["h"].get_embedding(data["batch_t"])

        t_i = self.embeddings["entity"]["t"].get_embedding(data["batch_h"])
        t_j = self.embeddings["entity"]["t"].get_embedding(data["batch_t"])

        r = self.embeddings["relation"]["r"].get_embedding(data["batch_r"])
        r_inv = self.embeddings["relation"]["r_inv"].get_embedding(data["batch_r"])

        score = self._calc_avg(h_i, t_i,h_j, t_j, r, r_inv).flatten()

        return score

    def predict(self, data):
        h = self.embeddings["entity"]["h"].get_embedding(data["batch_h"])
        t = self.embeddings["entity"]["t"].get_embedding(data["batch_t"])
        r = self.embeddings["relation"]["r"].get_embedding(data["batch_r"])
        
        score = -self._calc_ingr(h, r, t)
        return score



