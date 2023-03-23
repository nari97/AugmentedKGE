import torch
from Models.Model import Model


class RodE(Model):
    """
    Mojtaba Nayyeri, Mirza Mohtashim Alam, Jens Lehmann, Sahar Vahdati: 3D Learning and Reasoning in Link Prediction
        Over Knowledge Graphs. IEEE Access 8: 196459-196471 (2020).
    """
    def __init__(self, ent_total, rel_total, dim, variant='similarity'):
        """
            dim (int): Number of dimensions for embeddings
            variant can be either similarity or distance.
        """
        super(RodE, self).__init__(ent_total, rel_total)
        # We need to partition embeddings into chunks of size 3.
        self.dim = 3*(dim//3)
        self.variant = variant
        # We use the same as RotatE.
        self.pnorm = 1

    def get_default_loss(self):
        if self.variant == 'similarity':
            # "The multi-class logarithmic loss [...] with Nuclear 3-Norm Regularization is employed as loss function
            #   of Semantic-matching category."
            # Similar to GeomE, we are not differentiating between head and tail corruptions.
            return 'logsoftmax'
        if self.variant == 'distance':
            # "For Distance-based models, we use RotatE Loss function."
            return 'soft_margin'

    def get_score_sign(self):
        if self.variant == 'similarity':
            return 1
        if self.variant == 'distance':
            return -1

    def initialize_model(self):
        # Regularization is only used for the similarity variant. It suggests N3 regularization.
        self.create_embedding(self.dim, emb_type="entity", name="e", reg=self.variant == 'similarity')
        self.create_embedding(self.dim, emb_type="relation", name="r", reg=self.variant == 'similarity')
        # Section 3 discusses how to compute the optimal rotation. We include the rotation as a parameter.
        self.create_embedding(int(self.dim/3), emb_type="relation", name="theta", reg=self.variant == 'similarity')

        if self.variant == 'distance':
            self.create_embedding(self.dim, emb_type="relation", name="pr")
        
    def _calc(self, h, r, pr, theta, t):
        batch_size = h.shape[0]
        scores = torch.zeros((batch_size, 1), dtype=h.dtype, layout=h.layout, device=h.device)

        # We take slices of size 3.
        for i in range(0, self.dim, 3):
            h_slice, r_slice, t_slice = h[:, i:i+3], r[:, i:i+3], t[:, i:i+3]
            if self.variant == 'distance':
                pr_slice = pr[:, i:i+3]
            # There is only one theta per slice.
            theta_slice = theta[:, int(i/3)]

            # Let's compute R_r using Rodrigues's formula (Eq. (8)).
            K = torch.zeros((batch_size, 3, 3), dtype=h.dtype, layout=h.layout, device=h.device)
            K[:, 0, 1] = -r_slice[:, 2]
            K[:, 0, 2] = r_slice[:, 1]
            K[:, 1, 0] = r_slice[:, 2]
            K[:, 1, 2] = -r_slice[:, 0]
            K[:, 2, 0] = -r_slice[:, 1]
            K[:, 2, 1] = r_slice[:, 0]

            I = torch.ones((batch_size, 3, 3), dtype=h.dtype, layout=h.layout, device=h.device)

            R = torch.cos(theta_slice).view(batch_size, 1, 1) * I + \
                torch.sin(theta_slice).view(batch_size, 1, 1) * K + \
                (1 - torch.cos(theta_slice)).view(batch_size, 1, 1) * torch.matmul(K, K)

            h_times_r = torch.bmm(h_slice.view(batch_size, 1, 3), R).view(batch_size, 3)
            if self.variant == 'similarity':
                # Eq. (10).
                scores += torch.sum(h_times_r * t_slice, -1).view(-1, 1)

            if self.variant == 'distance':
                # Eq. (12).
                scores += torch.linalg.norm(h_times_r + pr_slice - t_slice, dim=-1, ord=self.pnorm).view(-1, 1)

        return scores

    def return_score(self, is_predict=False):
        (head_emb, rel_emb, tail_emb) = self.current_batch

        h = head_emb["e"]
        t = tail_emb["e"]
        r, theta = rel_emb["r"], rel_emb["theta"]

        pr = None
        if self.variant == 'distance':
            pr = rel_emb["pr"]

        return self._calc(h, r, pr, theta, t)
