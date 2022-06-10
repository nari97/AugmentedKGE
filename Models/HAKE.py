import math
import torch
from Models.Model import Model


class HAKE(Model):

    def __init__(self, ent_total, rel_total, dim):
        super(HAKE, self).__init__(ent_total, rel_total)
        self.dim = dim

    def get_default_loss(self):
        return 'soft_margin'

    def initialize_model(self):
        self.create_embedding(self.dim, emb_type="entity", name="em")
        self.create_embedding(self.dim, emb_type="entity", name="ep", init="uniform", init_params=[0, 2 * math.pi])
        self.create_embedding(self.dim, emb_type="relation", name="rm")
        self.create_embedding(self.dim, emb_type="relation", name="rmprime")
        self.create_embedding(self.dim, emb_type="relation", name="rp", init="uniform", init_params=[0, 2 * math.pi])
        self.create_embedding(1, emb_type="global", name="lambda1")
        self.create_embedding(1, emb_type="global", name="lambda2")

        self.register_scale_constraint(emb_type="relation", name="rmprime")

    def _calc(self, hm, hp, rm, rp, rmp, tm, tp, l1, l2):
        # rm and rmp must always be positive, so we use abs.
        modulus_scores = torch.linalg.norm(hm*((1-torch.abs(rmp))/(torch.abs(rm)+torch.abs(rmp)))-tm, dim=-1, ord=2)
        phase_scores = torch.linalg.norm(torch.sin((hp+rp-tp)/2), dim=-1, ord=1)
        return -l1*modulus_scores - l2*phase_scores

    def return_score(self, is_predict=False):
        (head_emb, rel_emb, tail_emb) = self.current_batch

        hm, hp = head_emb["em"], head_emb["ep"]
        tm, tp = tail_emb["em"], tail_emb["ep"]
        rm, rp, rmp = rel_emb["rm"], rel_emb["rp"], rel_emb["rmprime"]

        l1 = self.current_global_embeddings["lambda1"].item()
        l2 = self.current_global_embeddings["lambda2"].item()

        return self._calc(hm, hp, rm, rp, rmp, tm, tp, l1, l2)
