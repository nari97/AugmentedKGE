import time

import torch
from torch.autograd import Variable
import numpy as np
import math
from scipy.stats import wilcoxon
from tqdm import tqdm


class Materializer(object):
    """ This can be either the validator or tester depending on the manager used. E"""

    def __init__(self, manager=None, use_gpu=False, neg_predict=False):
        self.manager = manager
        self.use_gpu = use_gpu
        self.neg_predict = neg_predict

    def write_materialization_stats(self, f_materialization_stats, mat):

        batch_size = 10000
        n_batches = int(len(mat) / batch_size) + 1
        keys = list(mat.keys())
        values = list(mat.values())
        for batch in range(n_batches):
            end = (batch + 1) * batch_size

            if end > len(mat):
                end = len(mat)

            data_to_write = [
                str(keys[i][0]) + "\t" + str(keys[i][1]) + "\t" + str(keys[i][2]) + "\t" + str(values[i][0]) + "\t" + str(values[i][1]) + "\t" +
                str(values[i][2]) for i in range(batch * batch_size, end)]
            f_materialization_stats.write(
                '\n'.join(data_to_write) + '\n')

    def write_materializations(self, f_triples, mat):

        batch_size = 100000
        n_batches = int(len(mat) / batch_size) + 1
        for batch in range(n_batches):
            end = (batch + 1) * batch_size

            if end > len(mat):
                end = len(mat)
            data_to_write = [str(mat[i][0]) + "\t" + str(mat[i][1]) + "\t" + str(mat[i][2]) for i in range(batch * batch_size, end)]
            f_triples.write(
                '\n'.join(data_to_write) + '\n')
            del data_to_write

    def to_var(self, x, use_gpu):

        return Variable(torch.from_numpy(x))

    def materialize(self, model, name=None):

        collector = RankCollector()

        ctr = 0
        # We will split the evaluation by relation
        relations = {}
        for t in self.manager.get_triples():
            if t.r not in relations.keys():
                relations[t.r] = []
            relations[t.r].append(t)

        # f_materialization_stats = open(name + '_materialization_stats.tsv', 'w+')
        # f_materialized_triples = open(name + "_materialized.tsv", 'w+')
        # f_mispredicted_triples = open(name + "_mispredicted.tsv", "w+")

        positives_before_expected = {}
        bar = tqdm(total=self.manager.tripleTotal)

        total_negatives = 0
        total_materialized = 0
        total_mispredicted = 0
        for r in relations.keys():

            negative_ratio_dict = {}
            materialized_triples_list = []
            mispredicted_triples_list = []
            ctr += 1
            neg_triples = {}
            positives_before_expected[r] = 0

            for t in relations[r]:
                corruptedHeads = self.manager.get_corrupted(t.h, t.r, t.t, "head")
                corruptedTails = self.manager.get_corrupted(t.h, t.r, t.t, "tail")

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

                scores = self.predict(arrH, arrR, arrT, model).detach()

                if self.use_gpu:
                    scores = scores.cpu()

                rankhLess, ranktLess, rankhEq, ranktEq = 0, 0, 1, 1

                cHeads = scores[1:corruptedHeadsEnd]
                cTails = scores[corruptedHeadsEnd:]

                rankhLess = torch.sum(scores[0] > cHeads)
                ranktLess = torch.sum(scores[0] > cTails)
                rankhEq += torch.sum(scores[0] == cHeads)
                ranktEq += torch.sum(scores[0] == cTails)

                rankH = self.frac_rank(rankhLess, rankhEq)
                rankT = self.frac_rank(ranktLess, ranktEq)

                expectedH = (len(corruptedHeads) + 1) / 2
                expectedT = (len(corruptedTails) + 1) / 2

                if rankH < expectedH:
                    positives_before_expected[r] = positives_before_expected[r] + 1
                if rankT < expectedT:
                    positives_before_expected[r] = positives_before_expected[r] + 1

                for i in range(1, totalTriples):
                    if scores[0] >= scores[i]:
                        self.add_triple(neg_triples, arrH[i], arrR[i], arrT[i], 2)

                        if i < corruptedHeadsEnd:
                            if rankH < expectedH:
                                self.add_triple(neg_triples, arrH[i], arrR[i], arrT[i], 1)
                        else:
                            if rankT < expectedT:
                                self.add_triple(neg_triples, arrH[i], arrR[i], arrT[i], 1)

                collector.update_rank(self.frac_rank(rankhLess, rankhEq), rankhEq > 1, len(corruptedHeads),
                                      self.frac_rank(ranktLess, ranktEq), ranktEq > 1, len(corruptedTails), t.r,
                                      self.manager.relation_anomaly[t.r])
                bar.update(1)



            for key, value in neg_triples.items():
                # neg_triples position meanings
                # 0th position shows how many times g  enerated
                # 1st position shows how many times materialized (Ranked better than positive & positive below expected rank)
                # 2nd position shows how many times ranked better than positive

                negative_ratio_dict[(key[0], r, key[1])] = [value[0], value[1], value[2]]
                total_negatives += 1
                if value[1] > 0:
                    materialized_triples_list.append((key[0], r, key[1]))
                    total_materialized += 1
                if value[2] > 0:
                    mispredicted_triples_list.append((key[0], r, key[1]))
                    total_mispredicted += 1

            # self.write_materialization_stats(f_materialization_stats, negative_ratio_dict)
            # self.write_materializations(f_materialized_triples, materialized_triples_list)
            # self.write_materializations(f_mispredicted_triples, mispredicted_triples_list)

            break
        bar.close()

        # f_positives_before_expected = open(name + '_positives_before_expected.tsv', 'w+')
        print("Total negatives generated:", total_negatives)
        print("Total triples materialized:", total_materialized)
        print("Total triples materialized with mispredictions:", total_mispredicted)
        print("Writing files")

        # f_positives_before_expected.write(
        #     '\n'.join(
        #         (str(key) + "\t" + str(positives_before_expected[key])) for key in positives_before_expected.keys()))
        #
        # f_positives_before_expected.close()
        # f_materialization_stats.close()
        # f_materialized_triples.close()
        # f_mispredicted_triples.close()

        return collector

    def add_triple(self, tree, h, r, t, i):
        if (h, t) not in tree.keys():
            tree[(h, t)] = np.array((0, 0, 0))
        tree[(h, t)][i] = tree[(h, t)][i] + 1

    def frac_rank(self, less, eq):
        ret = 0
        # for i in range(eq):
        #     ret = ret + (less + (i + 1))
        # ret = ret / eq

        # TODO Change this if it works properly.
        ret = (2 * less + eq + 1) / 2
        # if otherRet != ret:
        #     print('The other calculation for fractional ranks did not work!!!!!!!!')
        return ret

    def predict(self, arrH, arrR, arrT, model):

        pred = model.predict({
            'batch_h': self.to_var(arrH, self.use_gpu),
            'batch_r': self.to_var(arrR, self.use_gpu),
            'batch_t': self.to_var(arrT, self.use_gpu),
            'mode': 'normal'
        })

        if self.neg_predict:
            pred = -pred
        return pred

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
