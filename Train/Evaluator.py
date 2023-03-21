import heapq
import pickle
import torch
import numpy as np
import math
#import time
from scipy.stats import wilcoxon


def add_triple(triple_stats, head, tail, is_negative=False, is_ranked_better=False, is_correctly_predicted=False,
               is_top_k=False):
    """
    Adds information about a triple (head, tail) to the triple_stats dictionary.

    Args:
        triple_stats (dict): A dictionary that maps pairs of entities (heads and tails) to information
            about triples involving those entities.
        head: The head entity of the triple.
        tail: The tail entity of the triple.
        is_negative (bool, optional): Whether the triple is negative (i.e., not true). Defaults to False.
        is_ranked_better (bool, optional): Whether the triple is ranked better than the correct answer by the model.
            Defaults to False.
        is_correctly_predicted (bool, optional): Whether the triple is correctly predicted by the model. Defaults to False.
        is_top_k (bool, optional): Whether the triple is in the top k predictions of the model. Defaults to False.

    Returns:
        None
    """
    index = (head, tail)

    # Index 0 is is_negative
    # Index 1 is ranked_better
    # Index 2 is correctly predicted
    # Index 3 is top-k

    if index not in triple_stats:
        triple_stats[index] = [0, 0, 0, 0]

    if is_negative:
        triple_stats[index][0] = 1

    if is_ranked_better:
        triple_stats[index][1] = 1

    if is_correctly_predicted:
        triple_stats[index][2] = 1

    if is_top_k:
        triple_stats[index][3] = 1


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

    def evaluate(self, model, materialize=False, materialize_basefile=None, k=25):
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
            n_positives_ranked_before_expected[r] = 0
            total += 1
            # start = time.perf_counter()
            # print("Relation", r, ":", total, "out of", len(relations))
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

                # Here I changed arrH, arrR and arrT to be a numpy matrix of size (totalTriples, 4), where positions
                # 0,1 and 2 are head, relation and tail respectively. The 4th position is to be used later when
                # the scores are calculated. This is so that the sorting and materialization process is easier to be
                # done
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

                # Adding scores to the triples
                triples[:, 3] = scores.detach().numpy()
                cHeads = triples[1:corruptedHeadsEnd, 3]
                cTails = triples[corruptedHeadsEnd:, 3]
                positive_triple_score = triples[0, 3]

                # Changed PyTorch.sum to numpy.sum as positive_triple_score, cHeads, cTails are all numpy arrays now.
                # We are expecting positives to have higher scores than negatives, so we need to add to the rank when
                #   the negatives scores are higher.
                rankhLess, ranktLess = np.sum(positive_triple_score < cHeads).item(), np.sum(
                    positive_triple_score < cTails).item()
                rankhEq, ranktEq = 1 + np.sum(positive_triple_score == cHeads).item(), 1 + np.sum(
                    positive_triple_score == cTails).item()

                # If it is NaN, rank last!
                if np.isnan(positive_triple_score.item()):
                    is_nan_cnt += 1
                    rankhLess, ranktLess = len(cHeads), len(cTails)
                    rankhEq, ranktEq = 1, 1

                if materialize:
                    # Compute ranks and expected rank
                    # TODO fractional ranks are computed again below.
                    # TODO Expected rank is computed in the collector.
                    rankH = self.frac_rank(rankhLess, rankhEq)
                    rankT = self.frac_rank(ranktLess, ranktEq)
                    expectedH = (len(corruptedHeads) + 1) / 2
                    expectedT = (len(corruptedTails) + 1) / 2

                    # Update dict, renamed the variable to make it more readable
                    if rankH < expectedH:
                        n_positives_ranked_before_expected[r] = n_positives_ranked_before_expected[r] + 1
                    if rankT < expectedT:
                        n_positives_ranked_before_expected[r] = n_positives_ranked_before_expected[r] + 1

                    # Get top-k triples for head and tail corruptions separately
                    top_k_triples_head = get_top_k_triples(triples[0:corruptedHeadsEnd], k=k)
                    top_k_triples_tail = get_top_k_triples(np.vstack((triples[0, :], triples[corruptedHeadsEnd:, :])),
                                                           k=k)

                    # Update top-k head triples while making sure that if the positive triple exists in top-k, then its
                    # is_negative is set to False, else set to True
                    for k_triple in top_k_triples_head:
                        if k_triple[0].item() == triples[0][0].item() and k_triple[1].item() == triples[0][1].item() and \
                                k_triple[2].item() == triples[0][2].item():
                            add_triple(triple_stats, k_triple[0], k_triple[2], is_top_k=True, is_negative=False)
                        else:
                            add_triple(triple_stats, k_triple[0], k_triple[2], is_top_k=True, is_negative=True)

                    # Update top-k tail triples while making sure that if the positive triple exists in top-k, then its
                    # is_negative is set to False, else set to True
                    for k_triple in top_k_triples_tail:
                        if k_triple[0].item() == triples[0][0].item() and k_triple[1].item() == triples[0][1].item() and \
                                k_triple[2].item() == triples[0][2].item():
                            add_triple(triple_stats, k_triple[0], k_triple[2], is_top_k=True, is_negative=False)
                        else:
                            add_triple(triple_stats, k_triple[0], k_triple[2], is_top_k=True, is_negative=True)

                    for i in range(1, totalTriples):
                        if positive_triple_score <= triples[i, 3]:

                            # Same as before now adding more information for each triples
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

                    # Since we have an incredible amount of negatives, I only want to keep the ones we need, i.e, the ones
                    # that were part of top-k or the ones that were ranked better (predictions)
                    if triple_stats[key][1] == 1 or triple_stats[key][3] == 1:
                        triple_stats_across_relations[(key[0], r, key[1])] = triple_stats[key]

            # end = time.perf_counter()
            # print('Took:', end-start)

        if materialize:
            for key in triple_stats_across_relations.keys():
                # Moved this to happen outside the testing loop
                if triple_stats_across_relations[key][1] == 1:
                    collector.update_unique_materialized(key[1])

            # Save positives_before_expected and triple_stats_across_relations into pickle files
            ranked_before_expected_file = open(f"{materialize_basefile}_ranked_before_expected.pickle", 'wb')
            triple_stats_file = open(f"{materialize_basefile}_triple_stats.pickle", 'wb')
            pickle.dump(n_positives_ranked_before_expected, ranked_before_expected_file)
            pickle.dump(triple_stats_across_relations, triple_stats_file)
            triple_stats_file.close()
            ranked_before_expected_file.close()

        # add_triple moved outside evaluator
        return collector

    def add_triple(self, tree, h, r, t, i):
        if (h, t) not in tree.keys():
            tree[(h, t)] = np.array((0, 0))
        tree[(h, t)][i] = tree[(h, t)][i] + 1

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


class RankCollector:
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

    # TODO This method and 'get_expected' are doing the same.
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
        cmp = 'low'
        if metric_str == 'mr':
            value = np.sum(ranks) / len(totals)
        elif metric_str == 'amr':
            value = 1.0 - (self.get(ranks, totals, 'mr').get() / self.get_expected(metric_str='mr').get())
            cmp = 'high'
        elif metric_str == 'wmr':
            # TODO Can this be done using Numpy?
            value, divisor = 0, 0
            for i in range(len(ranks)):
                value += totals[i] * ranks[i]
                divisor += totals[i]
            value = value / divisor
        elif metric_str == 'gmr':
            a = np.log(ranks)
            value = np.exp(a.sum() / len(a))
        elif metric_str == 'wgmr':
            # TODO Can this be done using Numpy?
            value, divisor = 0, 0
            for i in range(len(ranks)):
                value += totals[i] * math.log(ranks[i])
                divisor += totals[i]
            value = math.exp(value / divisor)
        elif metric_str == 'matsize':
            value = np.sum(ranks)
        if metric_str == 'mrr':
            # TODO Can this be done using Numpy?
            value = 0
            for i in range(len(ranks)):
                value += 1/ranks[i]
            value = value / len(totals)
            cmp = 'high'
        return Metric(value, cmp)


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
