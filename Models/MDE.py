import torch
from Models.Model import Model


class MDE(Model):
    def __init__(self, ent_total, rel_total, dim, norm=2):
        super(MDE, self).__init__(ent_total, rel_total)
        self.dim = dim
        self.pnorm = norm

    def get_default_loss(self):
        return 'limit'

    def initialize_model(self):
        for c in ['i', 'j', 'k', 'l']:
            self.create_embedding(self.dim, emb_type="entity", name="e_"+c)
            self.create_embedding(self.dim, emb_type="relation", name="r_"+c)
            self.create_embedding(1, emb_type="global", name="w_"+c)
        self.create_embedding(1, emb_type="global", name="psi")

    def si(self, w, h, r, t):
        return w*-torch.linalg.norm(h + r - t, dim=-1, ord=self.pnorm)

    def sj(self, w, h, r, t):
        return w*-torch.linalg.norm(t + r - h, dim=-1, ord=self.pnorm)

    def sk(self, w, h, r, t):
        return w*-torch.linalg.norm(h + t - r, dim=-1, ord=self.pnorm)

    def sl(self, w, h, r, t):
        return w*-torch.linalg.norm(h - r * t, dim=-1, ord=self.pnorm)

    def _calc(self, w, h, r, t, psi):
        (wi, wj, wk, wl) = w
        (hi, hj, hk, hl) = h
        (ri, rj, rk, rl) = r
        (ti, tj, tk, tl) = t
        return self.si(wi, hi, ri, ti) + self.sj(wj, hj, rj, tj) + self.sk(wk, hk, rk, tk) + \
                    self.sl(wl, hl, rl, tl) + psi

    def return_score(self, is_predict=False):
        (head_emb, rel_emb, tail_emb) = self.current_batch

        (hi, hj, hk, hl) = (head_emb["e_i"], head_emb["e_j"], head_emb["e_k"], head_emb["e_l"])
        (ti, tj, tk, tl) = (tail_emb["e_i"], tail_emb["e_j"], tail_emb["e_k"], tail_emb["e_l"])
        (ri, rj, rk, rl) = (rel_emb["r_i"], rel_emb["r_j"], rel_emb["r_k"], rel_emb["r_l"])
        (wi, wj, wk, wl) = (self.current_global_embeddings["w_i"], self.current_global_embeddings["w_j"],
                                self.current_global_embeddings["w_k"], self.current_global_embeddings["w_l"])
        psi = self.current_global_embeddings["psi"]

        return self._calc((wi, wj, wk, wl), (hi, hj, hk, hl), (ri, rj, rk, rl), (ti, tj, tk, tl), psi)
