import torch
import numpy as np
import math
from scipy.stats import wilcoxon
from Utils import DeviceUtils


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

    def evaluate(self, model, materialize=False):
        collector = RankCollector()

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
            total += 1
            # print (str(r) + ":Started")
            if materialize:
                neg_triples = {}
            for t in relations[r]:
                corruptedHeads = self.manager.get_corrupted(t.h, t.r, t.t, "head")
                corruptedTails = self.manager.get_corrupted(t.h, t.r, t.t, "tail")

                if materialize:
                    for hp in corruptedHeads:
                        self.add_triple(neg_triples, hp, t.r, t.t, 0)

                    for tp in corruptedTails:
                        self.add_triple(neg_triples, t.h, t.r, tp, 0)

                totalTriples = 1 + len(corruptedHeads) + len(corruptedTails)
                arrH = np.zeros(totalTriples, dtype=np.int64)
                arrR = np.zeros(totalTriples, dtype=np.int64)
                arrT = np.zeros(totalTriples, dtype=np.int64)

                arrH[0], arrR[0], arrT[0] = t.h, t.r, t.t

                arrH[1:1 + len(corruptedHeads)] = list(corruptedHeads)
                arrR[1:1 + len(corruptedHeads)] = t.r
                arrT[1:1 + len(corruptedHeads)] = t.t

                corruptedHeadsEnd = 1 + len(corruptedHeads)

                arrH[1 + len(corruptedHeads):] = t.h
                arrR[1 + len(corruptedHeads):] = t.r
                arrT[1 + len(corruptedHeads):] = list(corruptedTails)

                if not self.batched:
                    scores = self.predict(arrH, arrR, arrT, model)
                else:
                    batch_size = 25
                    arrH_batches = np.array_split(arrH, batch_size)
                    arrR_batches = np.array_split(arrR, batch_size)
                    arrT_batches = np.array_split(arrT, batch_size)
                    scores = torch.tensor([],device=DeviceUtils.get_device())

                    for i in range(len(arrT_batches)):
                        batch_score = self.predict(arrH_batches[i], arrR_batches[i], arrT_batches[i], model)
                        scores = torch.concat((scores, batch_score))

                cHeads = scores[1:corruptedHeadsEnd]
                cTails = scores[corruptedHeadsEnd:]

                rankhLess, ranktLess = torch.sum(scores[0] > cHeads).item(), torch.sum(scores[0] > cTails).item()
                rankhEq, ranktEq = 1 + torch.sum(scores[0] == cHeads).item(), 1 + torch.sum(scores[0] == cTails).item()

                # If it is NaN, rank last!
                if np.isnan(scores[0].item()):
                    is_nan_cnt += 1
                    rankhLess, ranktLess = len(cHeads), len(cTails)
                    rankhEq, ranktEq = 1, 1

                if materialize:
                    for i in range(1, totalTriples):
                        if scores[0] >= scores[i]:
                            self.add_triple(neg_triples, arrH[i], arrR[i], arrT[i], 1)

                collector.update_rank(self.frac_rank(rankhLess, rankhEq), rankhEq > 1, len(corruptedHeads),
                                      self.frac_rank(ranktLess, ranktEq), ranktEq > 1, len(corruptedTails), t.r,
                                      self.manager.relation_anomaly[t.r])

            if materialize:
                for p in neg_triples.keys():
                    collector.update_total_unique_triples(r)

                    if neg_triples[p][1] > 0:
                        collector.update_unique_materialized(r)

        # TODO Remove!
        print('IsNaN (%):', is_nan_cnt / total)
        return collector

    def add_triple(self, tree, h, r, t, i):
        if (h, t) not in tree.keys():
            tree[(h, t)] = np.array((0, 0))
        tree[(h, t)][i] = tree[(h, t)][i] + 1

    def frac_rank(self, less, eq):
        return (2 * less + eq + 1) / 2

    def predict(self, arrH, arrR, arrT, model):
        def to_var(x):
            if DeviceUtils.use_gpu:
                return torch.LongTensor(x).cuda()
            else:
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
    def stop_train(self, previous):
        if previous is None:
            return False

        current_metric = self.get_metric()

        # TODO Remove!
        print('Stop training')
        print('Current metric: ', current_metric.get(), '; Previous metric: ', previous.get_metric().get())
        print('Previous is better than current: ', Metric.is_improved(current_metric, previous.get_metric()))
        try:
            print('Is significant: ', RankCollector.is_significant(self.all_ranks, previous.all_ranks))
        except ValueError as err:
            print('Is significant error: ', err)
        print('Expected: ', previous.get_expected().get())
        print('Expected is better than current: ', Metric.is_improved(current_metric, previous.get_expected()))
        try:
            print('Is significant expected: ', self.is_significant_expected())
        except ValueError as err:
            print('Is significant error: ', err)

        # If the current metric is improved by previous and it is significant.
        if Metric.is_improved(current_metric, previous.get_metric()) and \
                RankCollector.is_significant(self.all_ranks, previous.all_ranks):
            return True

        # If the current metric is improved by random and it is significant.
        if Metric.is_improved(current_metric, previous.get_expected()) and self.is_significant_expected():
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
