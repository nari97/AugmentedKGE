import torch
from Models.Model import Model
from Utils import DeviceUtils


class TuckER(Model):

    def __init__(self, ent_total, rel_total, dim_e, dim_r):
        """
        Args:
            ent_total (int): Total number of entities
            rel_total (int): Total number of relations
            dim_e (int): Number of dimensions for entity embeddings
            dim_r (int): Number of dimensions for relation embeddings
        """
        super(TuckER, self).__init__(ent_total, rel_total, 0, "transd")

        self.dim_e = dim_e
        self.dim_r = dim_r

        self.create_embedding(self.dim_e, emb_type="entity", name="e")
        self.create_embedding(self.dim_r, emb_type="relation", name="r")

        self.W = torch.nn.Parameter(
            torch.empty((self.dim_e, self.dim_r, self.dim_e), dtype=torch.float64, device=DeviceUtils.get_device()))
        torch.nn.init.xavier_uniform_(self.W.data)

    def _calc(self, h, r, t):
        bs = h.view(h.shape[0], 1, -1)
        bp = r.view(r.shape[0], 1, -1)
        bo = t.view(t.shape[0], 1, -1)

        W = self.W

        I1 = torch.matmul(bs, W.view(W.shape[0], -1))
        I1 = I1.view(I1.shape[0], bs.shape[1], W.shape[1], W.shape[2])

        I2 = torch.bmm(bp, I1.view(I1.shape[0], I1.shape[2], -1))
        I2 = I2.view(I2.shape[0], -1, bp.shape[1], I1.shape[3])

        I3 = torch.bmm(bo, I2.view(I2.shape[0], I2.shape[3], -1))
        I3 = I3.view(I2.shape[0], -1, I2.shape[2], bo.shape[1])

        return torch.sigmoid(I3.flatten())


    def return_score(self, head_emb, rel_emb, tail_emb, is_predict=False):
        h = head_emb["e"]

        t = tail_emb["e"]

        r = rel_emb["r"]

        return self._calc(h, r, t)
