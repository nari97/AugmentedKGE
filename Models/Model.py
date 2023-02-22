import torch
import torch.nn as nn
from Utils.Embedding import Embedding
import os


class Model(nn.Module):
    """
        Base class for all models to inherit
    """

    def __init__(self, ent_tot, rel_tot):
        """
            ent_tot (int): Total number of entites
            rel_tot (int): Total number of relations
        """
        super(Model, self).__init__()

        self.ent_tot = ent_tot
        self.rel_tot = rel_tot
        self.use_gpu = False

        self.epoch = 0

        # Global are embeddings that do not belong to any entity or relation.
        self.embeddings = {"entity": {}, "relation": {}, "global" : {}}
        # This provides normalization info for each embedding. The info is 'norm', where 'p' and 'dim' are expected,
        #   'rescaling', where 'a' and 'b' are expected, or None.
        self.embeddings_normalization = {"entity": {}, "relation": {}, "global" : {}}
        # Regularization for embeddings.
        self.embeddings_regularization = {"entity": {}, "relation": {}, "global": {}}
        # This is for complex numbers.
        self.embeddings_regularization_complex = {"entity": [], "relation": [], "global": []}
        # TODO This regularization is executed once for the current batch. Are we using this?
        self.onthefly_regularization = []

        self.ranks = None
        self.totals = None
        self.hyperparameters = None

        self.custom_constraints = []
        # These constraints scale the embedding norms, e.g., ||x||<=1.
        self.scale_constraints = []
        # These constraints are executed once for the current batch.
        self.onthefly_constraints = []

        self.current_batch = None
        self.current_data = None
        self.current_global_embeddings = {}

    # https://stackoverflow.com/questions/3596641/how-to-get-subclass-name-from-a-static-method-in-python
    @classmethod
    def get_model_name(cls):
        return cls.__name__.lower()

    # This method declares the embeddings and all the stuff.
    def initialize_model(self):
        raise NotImplementedError

    # This method provides the default loss function of the model.
    def get_default_loss(self):
        raise NotImplementedError

    # PyTorch, please, do your magic!
    def forward(self, data):
        return self.get_scores(data)

    def get_scores(self, data, is_predict=False):
        """
        Calculate the scores for the given batch of triples
            data (Tensor): Tensor containing the batch of triples for which scores are to be calculated.
            is_predict (Bool): whether we are predicting (no gradient computation is expected).

        Returns:
            score (Tensor): Tensor containing the scores for each triple.
        """
        head_emb = self.get_head_embeddings(data)
        rel_emb = self.get_relation_embeddings(data)
        tail_emb = self.get_tail_embeddings(data)
        self.get_global_embeddings()

        self.current_batch = (head_emb, rel_emb, tail_emb)
        self.current_data = data

        # GPU is currently not supported. Larger datasets cannot fit in GPU.
        #self.embeddings_to_gpu()
        scores = self.return_score(is_predict=is_predict).flatten()
        #self.embeddings_to_cpu()

        return scores

    # This method creates embeddings.
    def create_embedding(self, dim, emb_type=None, name=None, register=True,
                         init_method="xavier_uniform", init_params=[],
                         norm_method=None, norm_params={"p": 2, "dim": -1},
                         reg=False, reg_params={"norm": torch.linalg.norm, "dim": -1}):
        """
            dim (int): Number of dimensions for embeddings.
            emb_type (String): entity, relation, global, None.
            name (String): name of the embedding to be accessed.
            register (Bool): whether to register as a model parameter or not.
            init_method and init_params: to be used by Embedding.
            norm_method and norm_params: embedding normalization.
            reg and reg_params: whether to regularize embedding and information.
        """
        if emb_type == "entity":
            total = self.ent_tot
        elif emb_type == "relation":
            total = self.rel_tot
        elif emb_type == "global":
            total = 1
        else:
            raise Exception("Type of embedding must be relation or entity")

        seed = self.hyperparameters.get('seed', None)
        if seed is not None:
            # Some models do not learn anything if all embeddings are equally initialized.
            # We add the number of embeddings declared so far to break ties.
            seed += len(self.embeddings[emb_type])
        emb = Embedding(total, dim, emb_type, name, init_method, init_params, seed=seed)
        if register:
            self.embeddings[emb_type][name] = emb
            self.embeddings_normalization[emb_type][name] = {'method': norm_method, 'params': norm_params}
            self.register_parameter(emb_type + '_' + name, self.embeddings[emb_type][name].emb)

        # This registers a regularization term for this embedding.
        if reg:
            self.embeddings_regularization[emb_type][name] = reg_params

        return emb

    # This method computes the score using self.current_batch.
    def return_score(self):
        raise NotImplementedError

    def predict(self, data):
        return self.get_scores(data, is_predict=True)

    # This method is called by the Trainer before starting the batch and gradients.
    def start_batch(self):
        self.apply_normalization()

    # Normalize embeddings.
    def apply_normalization(self):
        for emb_type in self.embeddings:
            for name in self.embeddings[emb_type]:
                emb = self.embeddings[emb_type][name]
                norm_info = self.embeddings_normalization[emb_type][name]

                if norm_info['method'] == 'norm':
                    p = norm_info['params']['p']
                    dim = norm_info['params']['dim']
                    # This is in place.
                    emb.emb.data = torch.nn.functional.normalize(emb.emb.data, p, dim)
                elif norm_info['method'] == 'rescaling':
                    # We want this in-place.
                    with torch.no_grad():
                        a = norm_info['params'].get('a', None)
                        b = norm_info['params'].get('b', None)
                        emb.emb.clamp(min=a, max=b)
                elif norm_info['method'] is None:
                    pass
                else:
                    print('Warning! You specified a norm method not recognized: ', norm_info['method'],
                          '; are you sure about this?')
                    pass

    # We register a scale constraint we wish to apply.
    def register_scale_constraint(self, emb_type, name, p=2, z=1, ctype='le'):
        self.scale_constraints.append({'emb_type': emb_type, 'name': name, 'p': p, 'z': z, 'ctype': ctype})

    # This registers a custom constraint that is "uncommon." See TransH for an example.
    def register_custom_constraint(self, c):
        self.custom_constraints.append(c)

    # Apply the constraints to be added to the loss.
    def constraints(self, data):
        head_emb = self.get_head_embeddings(data)
        tail_emb = self.get_tail_embeddings(data)
        rel_emb = self.get_relation_embeddings(data)

        all_constraints = {'custom': self.custom_constraints, 'scale': self.scale_constraints,
                           'onthefly': self.onthefly_constraints}

        constraints = 0
        for key in all_constraints.keys():
            for c in all_constraints[key]:
                v = []
                if key is 'custom':
                    v.append(c(head_emb, rel_emb, tail_emb))
                elif key is 'scale':
                    if c['emb_type'] == 'entity':
                        v.append(Model.scale_constraint(head_emb[c['name']], c['p'], c['z'], c['ctype']))
                        v.append(Model.scale_constraint(tail_emb[c['name']], c['p'], c['z'], c['ctype']))
                    elif c['emb_type'] == 'relation':
                        v.append(Model.scale_constraint(rel_emb[c['name']], c['p'], c['z'], c['ctype']))
                elif key is 'onthefly':
                    v.append(c)

                for x in v:
                    constraints += torch.sum(x)

        return constraints

    # Applies the constraint ||x||_p<=z when ctype='le', and ||x||_p>=z when ctype='ge'.
    # Check, for instance, TransH and GTrans on how these are applied.
    # In Keras, these are implemented using clamp: https://keras.io/api/layers/constraints/
    @staticmethod
    def scale_constraint(emb, p=2, z=1, ctype='le'):
        if ctype == 'le':
            constraint = torch.pow(torch.linalg.norm(emb, dim=-1, ord=p), 2) - z
        elif ctype == 'ge':
            constraint = z - torch.pow(torch.linalg.norm(emb, dim=-1, ord=p), 2)
        return torch.maximum(constraint, torch.zeros_like(constraint))

    # This method is called when the batch is done.
    def end_batch(self):
        # Clear on-the-fly constraints.
        self.onthefly_constraints = []

    # TODO Do we need complex regularization? If not, remove!
    # This method registers a new embedding regularization term for complex numbers.
    #def register_complex_regularization(self, emb_type=None, name_real=None, name_img=None,
    #                                     reg_params={"norm": torch.linalg.norm, "dim": -1}):
    #    self.embeddings_regularization_complex[emb_type].append(
    #        {"real_part": name_real, "img_part": name_img, "params": reg_params})

    # Apply regularization.
    def regularization(self, data, reg_type='L2'):
        reg, total = 0, 0

        head_emb = self.get_head_embeddings(data)
        tail_emb = self.get_tail_embeddings(data)
        rel_emb = self.get_relation_embeddings(data)

        for etype in self.embeddings_regularization.keys():
            embeds_to_apply = []
            if etype is 'entity':
                embeds_to_apply.append(head_emb)
                embeds_to_apply.append(tail_emb)
            elif etype is 'relation':
                embeds_to_apply.append(rel_emb)
            #elif type is 'global':
                #TODO: What to add?

            for ename in self.embeddings_regularization[etype].keys():
                if type is 'global':
                    # TODO What to do with global!
                    raise NotImplementedError

                for e in embeds_to_apply:
                    reg_params = self.embeddings_regularization[etype][ename]
                    r = self.apply_individual_regularization(e[ename], reg_type, reg_params)
                    reg += torch.sum(r)
                    total += len(r)

        for etype in self.embeddings_regularization_complex.keys():
            embeds_to_apply = []
            if etype is 'entity':
                embeds_to_apply.append(head_emb)
                embeds_to_apply.append(tail_emb)
            elif etype is 'relation':
                embeds_to_apply.append(rel_emb)
            # elif type is 'global':
                # TODO: What to add?

            for d in self.embeddings_regularization_complex[etype]:
                if type is 'global':
                    # TODO What to do with global!
                    raise NotImplementedError

                for e in embeds_to_apply:
                    real_part, img_part = e[d["real_part"]], e[d["img_part"]]
                    reg_params = d["params"]
                    complex = torch.view_as_complex(torch.stack((real_part, img_part), dim=-1))
                    r = self.apply_individual_regularization(complex, reg_type, reg_params)
                    reg += torch.sum(r)
                    total += len(r)

        for (x, reg_params) in self.onthefly_regularization:
            r = self.apply_individual_regularization(x, reg_type, reg_params)
            reg += torch.sum(r)
            total += len(r)

        # TODO Clear here?
        # Clear on-the-fly regularization.
        self.onthefly_regularization = []

        if reg_type is 'L2':
            reg = 1/2 * reg
        elif reg_type is 'L3':
            reg = 1/3 * reg

        if total > 0:
            reg /= total

        return reg

    # This method applies regularization to the parameters.
    def apply_individual_regularization(self, x, reg_type, reg_params):
        if reg_type is 'L1':
            p = 1
            f = torch.abs
        elif reg_type is 'L2':
            p = 2
            f = torch.pow
        elif reg_type is 'L3':
            p = 3
            f = torch.pow

        # It can be fro, for instance.
        other_p = reg_params.get("p", None)
        if other_p is not None:
            prev_p = p
            p = other_p

        if "transform" in reg_params.keys():
            x = reg_params["transform"](x)
        r = reg_params["norm"](x, dim=reg_params["dim"], ord=p)

        if other_p is not None:
            p = prev_p

        if p == 1:
            r = f(r)
        else:
            r = f(r, p)

        return r

    # TODO What about this? It seems this is not used.
    def register_onthefly_regularization(self, x, reg_params={"norm": torch.linalg.norm, "dim": -1}):
        self.onthefly_regularization.append((x, reg_params))

    # These methods set and get the hyperparameters of the model.
    def set_hyperparameters(self, hyperparams):
        self.hyperparameters = hyperparams

    def get_hyperparameters(self):
        return self.hyperparameters

    # These methods save and load models.
    def load_checkpoint(self, path):
        dic = torch.load(os.path.join(path))
        self.epoch = dic.pop("epoch", None)
        self.embeddings = dic.pop("embeddings")
        self.ranks = dic.pop("ranks")
        self.totals = dic.pop("totals")
        self.hyperparameters = dic.pop("hyperparameters")

        self.eval()

    def save_checkpoint(self, path, epoch=0):
        to_save = {"embeddings": self.embeddings, "ranks": self.ranks, "totals": self.totals,
                   "epoch": epoch, "hyperparameters": self.hyperparameters}

        torch.save(to_save, path)




    def get_embedding(self, emb_type, name):
        return self.embeddings[emb_type][name]

    def get_global_embeddings(self):
        self.current_global_embeddings = {}
        for emb in self.embeddings["global"]:
            self.current_global_embeddings[emb] = self.embeddings["global"][emb].emb
        return self.current_global_embeddings

    def get_head_embeddings(self, data):
        head_embeddings = {}
        for emb in self.embeddings["entity"]:
            head_embeddings[emb] = self.embeddings["entity"][emb].get_embedding(data["batch_h"])
        return head_embeddings

    def get_tail_embeddings(self, data):
        tail_embeddings = {}
        for emb in self.embeddings["entity"]:
            tail_embeddings[emb] = self.embeddings["entity"][emb].get_embedding(data["batch_t"])
        return tail_embeddings

    def get_relation_embeddings(self, data):
        relation_embeddings = {}
        for emb in self.embeddings["relation"]:
            relation_embeddings[emb] = self.embeddings["relation"][emb].get_embedding(data["batch_r"])
        return relation_embeddings







    def set_use_gpu(self, use_gpu):
        self.use_gpu = use_gpu
        for key1 in self.embeddings:
            for key2 in self.embeddings[key1]:
                self.embeddings[key1][key2] = self.embeddings[key1][key2].to("cuda")

    def move_to_gpu(self, emb):
        return emb.to(torch.device('cuda'))

    def move_to_cpu(self, emb):
        return emb.to(torch.device('cpu'))

    def embeddings_to_gpu(self):
        # Move to GPU?
        if self.use_gpu:
            (head_emb, rel_emb, tail_emb) = self.current_batch
            for emb in [head_emb, tail_emb, rel_emb]:
                for name in emb.keys():
                    emb[name] = self.move_to_gpu(emb[name])
            self.current_batch = (head_emb, rel_emb, tail_emb)

            for name in self.current_global_embeddings.keys():
                self.current_global_embeddings[name] = self.move_to_gpu(self.current_global_embeddings[name])

    def embeddings_to_cpu(self):
        if self.use_gpu:
            (head_emb, rel_emb, tail_emb) = self.current_batch
            for emb in [head_emb, tail_emb, rel_emb]:
                for name in emb.keys():
                    emb[name] = self.move_to_cpu(emb[name])
            self.current_batch = None
            self.current_data = None

            for name in self.current_global_embeddings.keys():
                self.current_global_embeddings[name] = self.move_to_cpu(self.current_global_embeddings[name])
            self.current_global_embeddings = {}

            torch.cuda.empty_cache()