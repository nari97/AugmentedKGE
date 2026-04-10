import json

import torch
import numpy as np
import os


class TopKEvaluator(object):
    """ This can be either the validator or tester depending on the manager used. E"""

    def __init__(self, manager=None, rel_anomaly_max=.75, rel_anomaly_min=0, batched=False):
        self.manager = manager
        self.rel_anomaly_max = rel_anomaly_max
        self.rel_anomaly_min = rel_anomaly_min
        self.batched = batched

    def evaluate_top_k(self, model, to_file, k=100):
        is_nan_cnt, total = 0, 0
        # We will split the evaluation by relation
        relations = {}
        for t in self.manager.get_triples():
            # We will not consider these anomalies.
            if self.manager.relation_anomaly[t.r] < self.rel_anomaly_min or \
                    self.manager.relation_anomaly[t.r] > self.rel_anomaly_max:
                continue
            if t.r not in relations.keys():
                relations[t.r] = []
            relations[t.r].append(t)

        for r in relations.keys():
            print(f"Processing relation: {total}/{len(relations)}: {r}")
            total += 1
            for t in relations[r]:
                for corrupt_type in ['head', 'tail']:
                    corrupted = self.manager.get_corrupted(t.h, t.r, t.t, corrupt_type)

                    total_triples = 1 + len(corrupted)
                    triples = np.zeros((total_triples, 4))

                    # This is the positive.
                    triples[0, 0:3] = [t.h, t.r, t.t]

                    # Adding the relation.
                    triples[1:total_triples, 1] = t.r

                    if corrupt_type == 'head':
                        triples[1:total_triples, 0] = list(corrupted)
                        triples[1:total_triples, 2] = t.t
                        query_str = f'({t.h},{t.r},{t.t})-h'
                        corrupted_position = 0

                    if corrupt_type == 'tail':
                        triples[1:total_triples, 0] = t.h
                        triples[1:total_triples, 2] = list(corrupted)
                        query_str = f'({t.h},{t.r},{t.t})-t'
                        corrupted_position = 2

                    # Getting the scores.
                    scores = self.predict(
                        triples[:, 0], triples[:, 1], triples[:, 2], model).flatten()

                    # Adding scores to the triples
                    triples[:, 3] = scores.detach().numpy()

                    negatives_scores = triples[1:, 3]
                    positive_triple_score = triples[0, 3]

                    rankLess = np.sum(positive_triple_score < negatives_scores).item()
                    rankEq = 1 + np.sum(positive_triple_score == negatives_scores).item()

                    # If it is NaN, rank last!
                    if np.isnan(positive_triple_score.item()):
                        is_nan_cnt += 1
                        rankLess, rankEq = len(corrupted), 1

                    def frac_rank():
                        return (2 * rankLess + rankEq + 1) / 2

                    rank = frac_rank()

                    # Sort by score: Get the indices (descending) and reorder.
                    triples = triples[triples[:, 3].argsort()[::-1]]

                    top_entities, top_scores = triples[:k, corrupted_position], triples[:k, 3]
                    bottom_entities, bottom_scores = triples[-k:, corrupted_position], triples[-k:, 3]

                    json_result = {'q': query_str, 'r': rank, 'N': len(corrupted),
                                   'te': top_entities.tolist(), 'ts': top_scores.tolist(),
                                   'be': bottom_entities.tolist(), 'bs': bottom_scores.tolist()}

                    with open(to_file, mode='a') as file:
                        file.write(json.dumps(json_result) + '\n')

    def evaluate_ranks(self, model, ranks_file):
        is_nan_cnt, total = 0, 0

        # We will split the evaluation by relation
        relations = {}

        for t in self.manager.get_triples():
            # We will not consider these anomalies.
            if self.manager.relation_anomaly[t.r] < self.rel_anomaly_min or \
                    self.manager.relation_anomaly[t.r] > self.rel_anomaly_max:
                continue
            if t.r not in relations.keys():
                relations[t.r] = []
            relations[t.r].append(t)

        for r in relations.keys():
            print(f"Processing relation: {total}/{len(relations)}: {r}")
            total += 1

            for t in relations[r]:
                for corrupt_type in ['head', 'tail']:
                    corrupted = self.manager.get_corrupted(t.h, t.r, t.t, corrupt_type)

                    total_triples = 1 + len(corrupted)
                    triples = np.zeros((total_triples, 4))

                    # This is the positive.
                    triples[0, 0:3] = [t.h, t.r, t.t]

                    # Adding the relation.
                    triples[1:total_triples, 1] = t.r

                    if corrupt_type == 'head':
                        triples[1:total_triples, 0] = list(corrupted)
                        triples[1:total_triples, 2] = t.t
                        query_str = f'({t.h},{t.r},{t.t})-h'
                        corrupted_position = 0

                    if corrupt_type == 'tail':
                        triples[1:total_triples, 0] = t.h
                        triples[1:total_triples, 2] = list(corrupted)
                        query_str = f'({t.h},{t.r},{t.t})-t'
                        corrupted_position = 2

                    # Getting the scores.
                    scores = self.predict(
                        triples[:, 0], triples[:, 1], triples[:, 2], model).flatten()

                    # Adding scores to the triples
                    triples[:, 3] = scores.detach().numpy()

                    negatives_scores = triples[1:, 3]
                    positive_triple_score = triples[0, 3]

                    rankLess = np.sum(positive_triple_score < negatives_scores).item()
                    rankEq = 1 + np.sum(positive_triple_score == negatives_scores).item()

                    # If it is NaN, rank last!
                    if np.isnan(positive_triple_score.item()):
                        is_nan_cnt += 1
                        rankLess, rankEq = len(corrupted), 1

                    def frac_rank():
                        return (2 * rankLess + rankEq + 1) / 2

                    rank = frac_rank()

                    json_result = {'q': query_str, 'r': rank, 'N': len(corrupted)}

                    with open(ranks_file, mode='a') as file:
                        file.write(json.dumps(json_result) + '\n')

    def predict(self, batch_h, batch_r, batch_t, model):
        def to_var(x):
            return torch.LongTensor(x)

        return model.predict({
            'batch_h': to_var(batch_h),
            'batch_r': to_var(batch_r),
            'batch_t': to_var(batch_t),
            'mode': 'normal'
        })
