import torch
from Utils import QuaternionUtils
from Models.Model import Model


class HopfE(Model):
    """
    Anson Bastos, Kuldeep Singh, Abhishek Nadgeri, Saeedeh Shekarpour, Isaiah Onando Mulang', Johannes Hoffart: HopfE:
        Knowledge Graph Representation Learning using Inverse Hopf Fibrations. CIKM 2021: 89-99.
    """
    def __init__(self, ent_total, rel_total, dim, norm=2):
        """
            dim (int): Number of dimensions for embeddings
            norm (int): L1 or L2 norm. Default: 2. See: https://github.com/ansonb/HopfE/blob/master/codes/model.py#L784
        """
        super(HopfE, self).__init__(ent_total, rel_total)
        self.dim = dim
        self.pnorm = norm

    def get_default_loss(self):
        # Like RotatE. After Eq. (8).
        return 'soft_margin'

    def get_score_sign(self):
        # It is a distance (norm).
        return -1

    def initialize_model(self):

        for component in ['a', 'b', 'c', 'd']:
            # Section 6: "The parameters are initialized using the He initialization [13]."
            self.create_embedding(self.dim, emb_type="relation", name="r_"+component, init_method="kaiming_uniform")

            # Entities do not have real component. Check Theorem 4.1 and Table 1.
            if component != 'a':
                self.create_embedding(self.dim, emb_type="entity", name="eh_"+component, init_method="kaiming_uniform")
                self.create_embedding(self.dim, emb_type="entity", name="et_"+component, init_method="kaiming_uniform")

    def _calc(self, h, r, t):
        # Rotation (Eq. (5)).
        def rotation(e_d, r_d, r_d_inv):
            return QuaternionUtils.hamilton_product(QuaternionUtils.hamilton_product(r_d, e_d), r_d_inv)

        # Hopf fibration (Eq. (1)).
        def hopf_fib(x):
            (x_a, x_b, x_c, x_d) = x
            return (torch.pow(x_a, 2) + torch.pow(x_b, 2) - torch.pow(x_c, 2) - torch.pow(x_d, 2),
                    2 * (x_a * x_d + x_b * x_c), 2 * (x_b * x_d - x_a * x_c))

        # This is just an aux function to subtract the result of two Hopf fibrations, compute the norms of their
        #   components and add them.
        def fib_out(f_1, f_2):
            (f_1_x, f_1_y, f_1_z), (f_2_x, f_2_y, f_2_z) = f_1, f_2

            def n(x, y):
                return torch.linalg.norm(x - y, dim=-1, ord=self.pnorm)

            return n(f_1_x, f_2_x) + n(f_1_y, f_2_y) + n(f_1_z, f_2_z)

        # Get inverse.
        r_inv = QuaternionUtils.inverse(r)

        # Eq. (7).
        return .5 * (fib_out(hopf_fib(rotation(h, r, r_inv)), hopf_fib(t)) +
                     fib_out(hopf_fib(rotation(t, r_inv, r)), hopf_fib(h)))

    def return_score(self, is_predict=False):
        (head_emb, rel_emb, tail_emb) = self.current_batch

        h_b, h_c, h_d = head_emb["eh_b"], head_emb["eh_c"], head_emb["eh_d"]
        t_b, t_c, t_d = tail_emb["et_b"], tail_emb["et_c"], tail_emb["et_d"]
        r_a, r_b, r_c, r_d = rel_emb["r_a"], rel_emb["r_b"], rel_emb["r_c"], rel_emb["r_d"]
        h_a, t_a = torch.zeros_like(h_b), torch.zeros_like(t_b)

        return self._calc((h_a, h_b, h_c, h_d), (r_a, r_b, r_c, r_d), (t_a, t_b, t_c, t_d))
