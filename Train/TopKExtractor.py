import heapq
import pickle
import torch
import numpy as np


def sort_by_score(item):
    """
    Returns the score of the given item to be used for sorting.

    Args:
        item (ndarray): A numpy matrix containing a triple to be sorted. The score should be in the fourth element (index 3).

    Returns:
        The score of the triple to be used for sorting.
    """
    return -item[3]


def get_top_k_triples(triples, k):
    """
    Returns the top k triples with the highest scores.

    Args:
        triples (ndarray): A numpy array representing the triples to be ranked.
            Each triple should have a score in the fourth element (index 3) of the tuple or list.
        k (int): The number of top triples to return.

    Returns:
        A list of the top k triples with the highest scores, sorted in ascending order by score.
    """
    top_k_triples = heapq.nsmallest(k, triples, key=sort_by_score)

    return top_k_triples


class TopK(object):
    """ This can be either the validator or tester depending on the manager used. E"""

    def __init__(self, manager=None, rel_anomaly_max=.75, rel_anomaly_min=0, batched=False):
        self.manager = manager
        self.rel_anomaly_max = rel_anomaly_max
        self.rel_anomaly_min = rel_anomaly_min
        self.batched = batched

    def get_totals(self):
        totals = []
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
            for t in relations[r]:
                totals.append(len(self.manager.get_corrupted(t.h, t.r, t.t, "head")))
                totals.append(len(self.manager.get_corrupted(t.h, t.r, t.t, "tail")))

        return totals

    def evaluate_top_k(self, model, materialize_basefile, k=100):
        is_nan_cnt, total = 0, 0
        # We will split the evaluation by relation
        relations = {}
        top_k_triples = {}
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
                corruptedHeads = self.manager.get_corrupted(t.h, t.r, t.t, "head")
                corruptedTails = self.manager.get_corrupted(t.h, t.r, t.t, "tail")

                totalTriples = 1 + len(corruptedHeads) + len(corruptedTails)

                triples = np.zeros((totalTriples, 4))

                triples[0, 0:3] = [t.h, t.r, t.t]

                triples[1:1 + len(corruptedHeads), 0] = list(corruptedHeads)
                triples[1:1 + len(corruptedHeads), 1] = t.r
                triples[1:1 + len(corruptedHeads), 2] = t.t

                corruptedHeadsEnd = 1 + len(corruptedHeads)

                triples[1 + len(corruptedHeads):, 0] = t.h
                triples[1 + len(corruptedHeads):, 1] = t.r
                triples[1 + len(corruptedHeads):, 2] = list(corruptedTails)

                scores = self.predict(triples[:, 0], triples[:, 1], triples[:, 2], model)

                triples[:, 3] = scores.detach().numpy()

                top_k_triples_head = get_top_k_triples(triples[0:corruptedHeadsEnd], k=k)
                top_k_triples_tail = get_top_k_triples(np.vstack((triples[0, :], triples[corruptedHeadsEnd:, :])),
                                                       k=k)

                for index in range(len(top_k_triples_head)):
                    k_triple = top_k_triples_head[index]
                    triple_to_add = (int(k_triple[0]), int(r), int(k_triple[1]))
                    if triple_to_add in top_k_triples:
                        top_k_triples[triple_to_add] = min(top_k_triples[triple_to_add], index + 1)
                    else:
                        top_k_triples[triple_to_add] = index + 1

                for index in range(len(top_k_triples_tail)):
                    k_triple = top_k_triples_tail[index]
                    triple_to_add = (int(k_triple[0]), int(r), int(k_triple[1]))
                    if triple_to_add in top_k_triples:
                        top_k_triples[triple_to_add] = min(top_k_triples[triple_to_add], index + 1)
                    else:
                        top_k_triples[triple_to_add] = index + 1

        top_k_file = open(f"{materialize_basefile}_top_{k}.pickle", "wb")
        pickle.dump(top_k_triples, top_k_file)
        top_k_file.close()

    def predict(self, arrH, arrR, arrT, model):
        def to_var(x):
            return torch.LongTensor(x)

        return model.predict({
            'batch_h': to_var(arrH),
            'batch_r': to_var(arrR),
            'batch_t': to_var(arrT),
            'mode': 'normal'
        })
