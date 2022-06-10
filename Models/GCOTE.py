import torch
from Models.Model import Model


# TODO This takes forever; find a better way, if possible.
class GCOTE(Model):

    def __init__(self, ent_total, rel_total, dim, head_context, tail_context, k=3):
        super(GCOTE, self).__init__(ent_total, rel_total)
        self.dim = dim
        self.head_context = head_context
        self.tail_context = tail_context
        self.k = k

    def get_default_loss(self):
        return 'soft_margin'

    def initialize_model(self):
        for k in range(0, self.k):
            self.create_embedding(self.dim, emb_type="entity", name="e_"+str(k))
            self.create_embedding((self.dim, self.dim), emb_type="relation", name="m_"+str(k))
            self.create_embedding(self.dim, emb_type="relation", name="s_"+str(k))

            # A square matrix is full rank if and only if its determinant is nonzero.
            m_embed = self.get_embedding("relation", "m_" + (str(k))).emb
            for r in range(0, m_embed.shape[0]):
                if torch.linalg.det(m_embed[r]) == 0:
                    print('Warning: the matrix does not have full rank!')

    # Adapted from: https://github.com/legendongary/pytorch-gram-schmidt/blob/master/gram_schmidt.py
    def gram_schmidt(self, vv):
        def projection(u, v):
            return (v * u).sum(dim=-1).view(-1, 1) / (u * u).sum(dim=-1).view(-1, 1) * u

        nk = vv.size(1)
        uu = torch.zeros_like(vv, device=vv.device)
        uu[:, :, 0] += vv[:, :, 0]
        for k in range(1, nk):
            vk = vv[:, k]
            uk = 0
            for j in range(0, k):
                uj = vv[:, :, j]
                uk = uk + projection(uj, vk)
            uu[:, :, k] += vk - uk
        for k in range(nk):
            uk = vv[:, :, k]
            uu[:, :, k] += uk / uk.norm()
        return uu

    # Equation 2.
    def proj_t(self, s, m, e):
        return torch.bmm(torch.bmm(torch.diag_embed(torch.exp(s)), self.gram_schmidt(m)), e.view(-1, self.dim, 1))

    # Equation 4.
    def proj_h(self, s, m, e):
        return torch.bmm(torch.bmm(torch.diag_embed(torch.exp(-s)),
                                   torch.transpose(self.gram_schmidt(m), 1, 2)), e.view(-1, self.dim, 1))

    def _calc(self, h, m, s, t):
        batch_size = h[0].shape[0]

        # Compute distances.
        scores = torch.zeros(batch_size, device=h[0].device)
        for k in range(0, self.k):
            scores += torch.linalg.norm(self.proj_t(s[k], m[k], h[k]).view(-1, self.dim) - t[k], dim=-1, ord=2)
            scores += torch.linalg.norm(self.proj_h(s[k], m[k], t[k]).view(-1, self.dim) - h[k], dim=-1, ord=2)

        # Compute context.
        for k in range(0, self.k):
            context_head, context_tail = h[k], t[k]

            for i in range(0, batch_size):
                (head, tail) = (self.current_data['batch_h'][i].item(), self.current_data['batch_t'][i].item())

                total_tail = 0
                for rp in self.tail_context.keys():
                    if tail in self.tail_context[rp].keys():
                        for hp in self.tail_context[rp][tail]:
                            m_c = self.get_embedding("relation", "m_"+(str(k))).emb[rp].view(1, self.dim, self.dim)
                            s_c = self.get_embedding("relation", "s_" + (str(k))).emb[rp].view(1, self.dim)
                            h_c = self.get_embedding("entity", "e_" + (str(k))).emb[hp].view(1, self.dim)

                            context_tail[i] += self.proj_t(s_c, m_c, h_c).view(self.dim)
                            total_tail += 1

                context_tail[i] /= total_tail + 1

                total_head = 0
                for rp in self.head_context.keys():
                    if head in self.head_context[rp].keys():
                        for tp in self.head_context[rp][head]:
                            m_c = self.get_embedding("relation", "m_" + (str(k))).emb[rp].view(1, self.dim, self.dim)
                            s_c = self.get_embedding("relation", "s_" + (str(k))).emb[rp].view(1, self.dim)
                            t_c = self.get_embedding("entity", "e_" + (str(k))).emb[tp].view(1, self.dim)

                            context_head[i] += self.proj_h(s_c, m_c, t_c).view(self.dim)
                            total_head += 1

                context_head[i] /= total_head + 1

            scores += torch.linalg.norm(self.proj_t(s[k], m[k], h[k]).view(-1, self.dim) - context_tail, dim=-1, ord=2)
            scores += torch.linalg.norm(self.proj_h(s[k], m[k], t[k]).view(-1, self.dim) - context_head, dim=-1, ord=2)

        # They are all distances, so negative.
        return -scores

    def return_score(self, is_predict=False):
        (head_emb, rel_emb, tail_emb) = self.current_batch

        h, m, s, t = [], [], [], []
        for k in range(0, self.k):
            h.append(head_emb["e_"+str(k)])
            t.append(tail_emb["e_"+str(k)])
            m.append(rel_emb["m_"+str(k)])
            s.append(rel_emb["s_"+str(k)])

        return self._calc(h, m, s, t)
