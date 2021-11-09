import torch
from torch.autograd import Variable
import numpy as np
import math
from scipy.stats import wilcoxon

class Evaluator(object):

    """ This can be either the validator or tester depending on the manager used. E"""
    def __init__(self, manager=None, use_gpu=False, rel_anomaly_max=.75, rel_anomaly_min=0):
        self.manager = manager
        self.use_gpu = use_gpu
        if self.use_gpu:
            self.model.cuda()
        self.rel_anomaly_max = rel_anomaly_max
        self.rel_anomaly_min = rel_anomaly_min

    def to_var(self, x, use_gpu):
        if use_gpu:
            return Variable(torch.from_numpy(x).cuda())
        else:
            return Variable(torch.from_numpy(x))

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

    def evaluate(self, model, materialize=False, name = None, dataset = None):
        collector = RankCollector()
        
        # rank_test_h = []
        # ranks_test_t = []
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
            #print (str(r) + ":Started")
            if materialize:
                neg_triples = {}
                #neg_counts = {}
                count = 0
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

                arrH[1:1+len(corruptedHeads)] = list(corruptedHeads)
                arrR[1:1+len(corruptedHeads)] = t.r
                arrT[1:1+len(corruptedHeads)] = t.t

                corruptedHeadsEnd = 1+len(corruptedHeads)

                arrH[1+len(corruptedHeads):] = t.h
                arrR[1+len(corruptedHeads):] = t.r
                arrT[1+len(corruptedHeads):] = list(corruptedTails)
                
                scores = self.predict(arrH, arrR, arrT, model)

                rankhLess, ranktLess, rankhEq, ranktEq = 0, 0, 1, 1

                cHeads = scores[1:corruptedHeadsEnd]
                cTails = scores[corruptedHeadsEnd:]
                
                rankhLess = torch.sum(scores[0]>cHeads).item()
                ranktLess = torch.sum(scores[0]>cTails).item()
                rankhEq += torch.sum(scores[0] == cHeads).item()
                ranktEq += torch.sum(scores[0] == cTails).item()
                
                if materialize:
                    for i in range(1, totalTriples):
                            if scores[0] >= scores[i]:
                                self.add_triple(neg_triples, arrH[i], arrR[i], arrT[i], 1)
                                #self.add_triple(neg_triples, arrH[i], arrR[i], arrT[i], 2)
                        
                            
                # rank_test_h.append(self.frac_rank(rankhLess, rankhEq))
                # ranks_test_t.append(self.frac_rank(ranktLess, ranktEq))
                

                collector.update_rank(self.frac_rank(rankhLess, rankhEq), rankhEq>1, len(corruptedHeads),
                           self.frac_rank(ranktLess, ranktEq), ranktEq>1, len(corruptedTails), t.r, self.manager.relation_anomaly[t.r])
  
            #print (str(r) + ":Materialized")    
            if materialize:
                count = 0
                for p in neg_triples.keys():
                    collector.update_total_unique_triples(r)
                    
                    if neg_triples[p][1] > 0:
                        collector.update_unique_materialized(r)
                        
                    

            #print (str(r) + ":Finished")

        
        
        # rank_test = []
        # for i in range(len(rank_test_h)):
        #     rank_test.append(rank_test_h[i])
        #     rank_test.append(ranks_test_t[i])

        # print ('Tested Ranks : ', len(rank_test), len(collector.all_ranks))

        # if len(rank_test) == len(collector.all_ranks):
        #     miss = 0
        #     for i in range(len(rank_test)):
        #         if rank_test[i]!=collector.all_ranks[i]:
        #             miss+=1

        #     print ('Miss : ', miss)
        # else:
        #     print ('Lengths are not the same!')

        return collector

    def add_triple(self, tree, h, r, t, i):
        if (h, t) not in tree.keys():
             tree[(h, t)] = np.array((0, 0))
        tree[(h, t)][i] = tree[(h, t)][i] + 1

    def frac_rank(self, less, eq):
        ret = 0
        for i in range(eq):
            ret = ret + (less + (i+1))
        ret = ret / eq

        # TODO Change this if it works properly.
        otherRet = (2*less + eq + 1)/2;
        if otherRet != ret:
            print('The other calculation for fractional ranks did not work!!!!!!!!')

        return ret

    def predict(self, arrH, arrR, arrT, model):
        return model.predict({
            'batch_h': self.to_var(arrH, self.use_gpu),
            'batch_r': self.to_var(arrR, self.use_gpu),
            'batch_t': self.to_var(arrT, self.use_gpu),
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
            below.append(self.all_ranks[i] > (self.all_totals[i]+1)/2)
        return below

    def update_unique_materialized(self, r):
        if r not in self.unique_triples_materialized.keys():
            self.unique_triples_materialized[r] = 0
        self.unique_triples_materialized[r] = self.unique_triples_materialized[r]+1

    def update_total_unique_triples(self, r):
        if r not in self.total_unique_triples.keys():
            self.total_unique_triples[r] = 0
        self.total_unique_triples[r] = self.total_unique_triples[r]+1

    def get_expected(self, metric_str="mrh"):
        expected=[]
        for i in range(len(self.all_totals)):
            expected.append((self.all_totals[i] + 1) / 2)
        return self.get(expected, self.all_totals, metric_str)

    def get_metric(self, metric_str="mrh"):
        return self.get(self.all_ranks, self.all_totals, metric_str)

    def is_significant(self, other_ranks, threshold=.05):
        return wilcoxon(self.all_ranks, other_ranks, zero_method='pratt').pvalue < threshold

    def is_significant_expected(self):
        expected = []
        for i in range(len(self.all_totals)):
            expected.append((self.all_totals[i] + 1) / 2)
        return self.is_significant(expected)

    def get(self, ranks, totals, metric_str):
        if len(ranks) == 0:
            return Metric(0)
        if metric_str == 'mr':
            value = 0
            for r in ranks:
                value = value + r
            value = value / len(totals)
        elif metric_str == 'mrg':
            a = np.log(ranks)
            value = np.exp(a.sum() / len(a))
        elif metric_str == 'mrh':
            value, divisor = 0, 0
            for i in range(len(ranks)):
                value = value + totals[i] * math.log(ranks[i])
                divisor = divisor + totals[i]
            value = math.exp(value / divisor)
        return Metric(value)

class Metric():
    def __init__(self, value, cmp='low'):
        self.value = value
        self.cmp = cmp

    # other is better than self
    def is_improved(self, other):
        if self.cmp == 'low':
            return self.get() > other.get()
        elif self.cmp == 'high':
            return self.get() < other.get()

    def get(self):
        return self.value
