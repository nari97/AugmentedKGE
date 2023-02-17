import heapq
import pickle

import pandas as pd
import torch
import numpy as np
import math
from scipy.stats import wilcoxon


def add_triple(triple_stats, head, tail, is_negative=False, is_ranked_better=False, is_correctly_predicted=False,
               is_top_k=False):
    index = (head, tail)

    if index not in triple_stats:
        triple_stats[index] = {'is_negative': False, 'is_ranked_better': False, 'is_correctly_predicted': False,
                               'is_top_k': False}

    if is_negative:
        triple_stats[index]['is_negative'] = True

    if is_ranked_better:
        triple_stats[index]['is_ranked_better'] = True

    if is_correctly_predicted:
        triple_stats[index]['is_correctly_predicted'] = True

    if is_top_k:
        triple_stats[index]['is_top_k'] = True


def sort_by_score(item):
    return item[3]


def get_top_k_triples(triples, k):
    top_k_triples = heapq.nsmallest(k, triples, key=sort_by_score)

    return top_k_triples


def save_positives_and_triple_stats(n_positives_ranked_before_expected, triple_stats, folder_to_save, model_name,
                                    dataset_name):
    ranked_before_expected_file = open(f"{folder_to_save}\\{dataset_name}_{model_name}_ranked_before_expected.pickle",
                                       'wb')
    triple_stats_file = open(f"{folder_to_save}\\{dataset_name}_{model_name}_triple_stats.pickle", 'wb')

    pickle.dump(n_positives_ranked_before_expected, ranked_before_expected_file)
    pickle.dump(triple_stats, triple_stats_file)

    triple_stats_file.close()
    ranked_before_expected_file.close()


class Evaluator(object):
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

    def evaluate(self, model, materialize=False, folder_to_save=None, model_name=None, dataset_name=None):
        collector = RankCollector()

        is_nan_cnt, total = 0, 0
        # We will split the evaluation by relation
        relations = {}
        triple_stats_across_relations = {}
        n_positives_ranked_before_expected = {}
        for t in self.manager.get_triples():
            # We will not consider these anomalies.
            if self.manager.relation_anomaly[t.r] < self.rel_anomaly_min or \
                    self.manager.relation_anomaly[t.r] > self.rel_anomaly_max:
                continue
            if t.r not in relations.keys():
                relations[t.r] = []
            relations[t.r].append(t)

        for r in relations.keys():
            total += 1
            # print (str(r) + ":Started")
            print(f"Relation {r}: {total}/{len(relations)}")
            if materialize:
                triple_stats = {}
            for t in relations[r]:
                corruptedHeads = self.manager.get_corrupted(t.h, t.r, t.t, "head")
                corruptedTails = self.manager.get_corrupted(t.h, t.r, t.t, "tail")

                if materialize:
                    for hp in corruptedHeads:
                        add_triple(triple_stats, hp, t.t, is_negative=True)

                    for tp in corruptedTails:
                        add_triple(triple_stats, t.h, tp, is_negative=True)

                totalTriples = 1 + len(corruptedHeads) + len(corruptedTails)

                triples = np.zeros((totalTriples, 4))

                # arrH = np.zeros(totalTriples, dtype=np.int64)
                # arrR = np.zeros(totalTriples, dtype=np.int64)
                # arrT = np.zeros(totalTriples, dtype=np.int64)

                triples[0, 0:3] = [t.h, t.r, t.t]
                # arrH[0], arrR[0], arrT[0] = t.h, t.r, t.t

                triples[1:1 + len(corruptedHeads), 0] = list(corruptedHeads)
                triples[1:1 + len(corruptedHeads), 1] = t.r
                triples[1:1 + len(corruptedHeads), 2] = t.t

                # arrH[1:1 + len(corruptedHeads)] = list(corruptedHeads)
                # arrR[1:1 + len(corruptedHeads)] = t.r
                # arrT[1:1 + len(corruptedHeads)] = t.t

                corruptedHeadsEnd = 1 + len(corruptedHeads)

                triples[1 + len(corruptedHeads):, 0] = t.h
                triples[1 + len(corruptedHeads):, 1] = t.r
                triples[1 + len(corruptedHeads):, 2] = list(corruptedTails)

                # arrH[1 + len(corruptedHeads):] = t.h
                # arrR[1 + len(corruptedHeads):] = t.r
                # arrT[1 + len(corruptedHeads):] = list(corruptedTails)

                if not self.batched:
                    scores = self.predict(triples[:, 0], triples[:, 1], triples[:, 2], model)
                else:
                    batch_size = 25
                    arrH_batches = np.array_split(triples[:, 0], batch_size)
                    arrR_batches = np.array_split(triples[:, 1], batch_size)
                    arrT_batches = np.array_split(triples[:, 2], batch_size)
                    scores = None

                    for i in range(len(arrT_batches)):
                        batch_score = self.predict(arrH_batches[i], arrR_batches[i], arrT_batches[i], model)
                        if scores is None:
                            scores = batch_score
                        else:
                            scores = torch.concat((scores, batch_score))

                triples[:, 3] = scores.detach().numpy()
                cHeads = triples[1:corruptedHeadsEnd, 3]
                cTails = triples[corruptedHeadsEnd:, 3]
                positive_triple_score = triples[0, 3]

                rankhLess, ranktLess = np.sum(positive_triple_score > cHeads).item(), np.sum(
                    positive_triple_score > cTails).item()
                rankhEq, ranktEq = 1 + np.sum(positive_triple_score == cHeads).item(), 1 + np.sum(
                    positive_triple_score == cTails).item()

                # If it is NaN, rank last!
                if np.isnan(positive_triple_score.item()):
                    is_nan_cnt += 1
                    rankhLess, ranktLess = len(cHeads), len(cTails)
                    rankhEq, ranktEq = 1, 1

                if materialize:
                    rankH = self.frac_rank(rankhLess, rankhEq)
                    rankT = self.frac_rank(ranktLess, ranktEq)
                    expectedH = (len(corruptedHeads) + 1) / 2
                    expectedT = (len(corruptedTails) + 1) / 2

                    if rankH < expectedH:
                        n_positives_ranked_before_expected[r] = n_positives_ranked_before_expected[r] + 1
                    if rankT < expectedT:
                        n_positives_ranked_before_expected[r] = n_positives_ranked_before_expected[r] + 1

                    top_k_triples = get_top_k_triples(triples, k=25)

                    for k_triple in top_k_triples:
                        if k_triple[0] == triples[0][0] and k_triple[1] == triples[1] and k_triple[2] == triples[2]:
                            add_triple(triple_stats, k_triple[0], k_triple[2], is_top_k=True, is_negative=False)
                        else:
                            add_triple(triple_stats, k_triple[0], k_triple[2], is_top_k=True, is_negative=True)

                    for i in range(1, totalTriples):
                        if positive_triple_score >= triples[i, 3]:
                            add_triple(triple_stats, triples[i, 0], triples[i, 2], is_ranked_better=True)
                            if i < corruptedHeadsEnd:
                                if rankH < expectedH:
                                    add_triple(triple_stats, triples[i, 0], triples[i, 2], is_correctly_predicted=True)
                            else:
                                if rankT < expectedT:
                                    add_triple(triple_stats, triples[i, 0], triples[i, 2], is_correctly_predicted=True)

                collector.update_rank(self.frac_rank(rankhLess, rankhEq), rankhEq > 1, len(corruptedHeads),
                                      self.frac_rank(ranktLess, ranktEq), ranktEq > 1, len(corruptedTails), t.r,
                                      self.manager.relation_anomaly[t.r])

            if materialize:
                for key in triple_stats.keys():
                    collector.update_total_unique_triples(r)
                    if triple_stats[key]['is_ranked_better'] or triple_stats[key]['is_top_k']:
                        triple_stats_across_relations[(key[0], r, key[1])] = triple_stats[key]

        if materialize:
            for key in triple_stats_across_relations.keys():
                if triple_stats_across_relations[key]['is_ranked_better']:
                    collector.update_unique_materialized(key[1])
            save_positives_and_triple_stats(n_positives_ranked_before_expected, triple_stats,
                                            folder_to_save=f"{folder_to_save}\\{dataset_name}\\{model_name}",
                                            model_name=model_name, dataset_name=dataset_name)

        return collector

    def frac_rank(self, less, eq):
        return (2 * less + eq + 1) / 2

    def predict(self, arrH, arrR, arrT, model):
        def to_var(x):
            return torch.LongTensor(x)

        return model.predict({
            'batch_h': to_var(arrH),
            'batch_r': to_var(arrR),
            'batch_t': to_var(arrT),
            'mode': 'normal'
        })


class RankCollector():
    def __init__(self):
        self.all_ranks = []
        self.all_totals = []
        self.all_rels = []
        self.all_anomalies = []
        self.all_ties = []
        self.unique_triples_materialized = {}
        self.total_unique_triples = {}

    def load(self, r, t):
        self.all_ranks = r
        self.all_totals = t

    def prune(self, max_anom, min_anom):
        rc = RankCollector()
        for i in range(len(self.all_anomalies)):
            if self.all_anomalies[i] < min_anom or self.all_anomalies[i] > max_anom:
                continue

            rc.all_ranks.append(self.all_ranks[i])
            rc.all_totals.append(self.all_totals[i])
            rc.all_rels.append(self.all_rels[i])
            rc.all_anomalies.append(self.all_anomalies[i])
            rc.all_ties.append(self.all_ties[i])

            if self.all_rels[i] in self.unique_triples_materialized.keys() \
                    and self.all_rels[i] not in rc.unique_triples_materialized.keys():
                rc.unique_triples_materialized[self.all_rels[i]] = self.unique_triples_materialized[self.all_rels[i]]
            if self.all_rels[i] in self.total_unique_triples.keys() \
                    and self.all_rels[i] not in rc.total_unique_triples.keys():
                rc.total_unique_triples[self.all_rels[i]] = self.total_unique_triples[self.all_rels[i]]
        return rc

    # Checks whether we should stop training.
    def stop_train(self, previous, metric_str="mr"):
        if previous is None:
            return False

        current_metric = self.get_metric(metric_str=metric_str)

        # If the current metric is improved by previous and it is significant.
        if Metric.is_improved(current_metric, previous.get_metric(metric_str=metric_str)) and \
                RankCollector.is_significant(self.all_ranks, previous.all_ranks):
            return True

        # If the current metric is improved by random and it is significant.
        if Metric.is_improved(current_metric, previous.get_expected(metric_str=metric_str)) and \
                self.is_significant_expected():
            return True

        return False

    def update_rank(self, rankh, hHasTies, totalh, rankt, tHasTies, totalt, r, anomaly):
        self.all_ranks.append(rankh)
        self.all_ties.append(hHasTies)
        self.all_ranks.append(rankt)
        self.all_ties.append(tHasTies)
        self.all_totals.append(totalh)
        self.all_totals.append(totalt)
        self.all_rels.append(r)
        self.all_rels.append(r)
        self.all_anomalies.append(anomaly)
        self.all_anomalies.append(anomaly)

    def get_ranks_below_expected(self):
        below = []
        for i in range(len(self.all_totals)):
            below.append(self.all_ranks[i] > (self.all_totals[i] + 1) / 2)
        return below

    def update_unique_materialized(self, r):
        if r not in self.unique_triples_materialized.keys():
            self.unique_triples_materialized[r] = 0
        self.unique_triples_materialized[r] = self.unique_triples_materialized[r] + 1

    def update_total_unique_triples(self, r):
        if r not in self.total_unique_triples.keys():
            self.total_unique_triples[r] = 0
        self.total_unique_triples[r] = self.total_unique_triples[r] + 1

    def get_expected(self, metric_str="mr"):
        expected = []
        for i in range(len(self.all_totals)):
            expected.append((self.all_totals[i] + 1) / 2)
        return self.get(expected, self.all_totals, metric_str)

    def get_metric(self, metric_str="mr"):
        return self.get(self.all_ranks, self.all_totals, metric_str)

    @staticmethod
    def is_significant(these_ranks, other_ranks, threshold=.05):
        return wilcoxon(these_ranks, other_ranks, zero_method='pratt').pvalue < threshold

    def is_significant_expected(self):
        expected = []
        for i in range(len(self.all_totals)):
            expected.append((self.all_totals[i] + 1) / 2)
        return RankCollector.is_significant(self.all_ranks, expected)

    # TODO Compute these using mean and include variance.
    def get(self, ranks, totals, metric_str):
        if len(ranks) == 0:
            return Metric(0)
        if metric_str == 'mr':
            value = np.sum(ranks) / len(totals)
        if metric_str == 'wmr':
            value, divisor = 0, 0
            for i in range(len(ranks)):
                value += totals[i] * ranks[i]
                divisor += totals[i]
            value = value / divisor
        elif metric_str == 'gmr':
            a = np.log(ranks)
            value = np.exp(a.sum() / len(a))
        elif metric_str == 'wgmr':
            value, divisor = 0, 0
            for i in range(len(ranks)):
                value += totals[i] * math.log(ranks[i])
                divisor += totals[i]
            value = math.exp(value / divisor)
        return Metric(value)


class Metric():
    def __init__(self, value, cmp='low'):
        self.value = value
        self.cmp = cmp

    # Check whether this is improved by that.
    @staticmethod
    def is_improved(this, that):
        if this.cmp != that.cmp:
            raise ValueError('Comparison types of this (' + this.cmp + ') and that (' + that.cmp + ') are different')
        if this.cmp == 'low':
            return this.get() > that.get()
        elif this.cmp == 'high':
            return this.get() < that.get()

    def get(self):
        return self.value
