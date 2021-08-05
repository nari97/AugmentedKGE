import numpy as np
import random
import sys
from .DataLoader import DataLoader

class DataSampler(object):

    def __init__(self, nbatches, datasampler):
        self.nbatches = nbatches
        self.datasampler = datasampler
        self.batch = 0

    def __iter__(self):
        return self

    def __next__(self):
        self.batch += 1
        if self.batch > self.nbatches:
            raise StopIteration()
        return self.datasampler()

    def __len__(self):
        return self.nbatches


class TripleManager():
    """
    splits contains a list of the splits to consider. Usually, ['train'] for training, ['validation', 'train'] for validation,
    and ['test', 'validation', 'train'] for test. The order is important, the first position must be the main one.
    """
    def __init__(self, path, splits, nbatches=None, neg_ent=None, neg_rel=None, use_bern=False, seed=None, corruption_mode="Global"):
        self.counter = 0
        # Whether we will use a Bernoulli distribution to determine whether to corrupt head or tail
        self.use_bern = use_bern

        loaders = []
        self.tripleList = []
        headEntities, tailEntities = set(), set()
        self.headDict, self.tailDict, dom, ran = {}, {}, {}, {}

        for s in splits:
            loader = DataLoader(path, s)
            self.entityTotal = loader.entityTotal
            self.relationTotal = loader.relationTotal

            loaders = loaders + [loader]
            if len(self.tripleList) == 0:
                self.tripleList = loader.getTriples()
            headEntities = headEntities.union(loader.getHeadEntities())
            tailEntities = tailEntities.union(loader.getTailEntities())

            for r in loader.relations:
                if r in loader.getHeadDict():
                    if r not in self.headDict:
                        self.headDict[r] = {}
                    for t in loader.getHeadDict()[r]:
                        if t not in self.headDict[r]:
                            self.headDict[r][t] = set()
                        self.headDict[r][t] = self.headDict[r][t].union(loader.getHeadDict()[r][t])

                if r in loader.getTailDict():
                    if r not in self.tailDict:
                        self.tailDict[r] = {}
                    for h in loader.getTailDict()[r]:
                        if h not in self.tailDict[r]:
                            self.tailDict[r][h] = set()
                        self.tailDict[r][h] = self.tailDict[r][h].union(loader.getTailDict()[r][h])

                if r in loader.getDomain():
                    if r not in dom:
                        dom[r] = set()
                    dom[r] = dom[r].union(loader.getDomain()[r])

                if r in loader.getRange():
                    if r not in ran:
                        ran[r] = set()
                    ran[r] = ran[r].union(loader.getRange()[r])

        self.nbatches = nbatches
        self.negative_ent = neg_ent
        self.negative_rel = neg_rel

        if self.use_bern:
            # tph: the average number of tail entities per head entity
            # hpt: the average number of head entities per tail entity
            tph, hpt = {}, {}
            relations = set()
            for r in self.tailDict:
                tph[r] = 0
                for h in self.tailDict[r]:
                    tph[r] += len(self.tailDict[r][h])
                tph[r] = tph[r]/len(self.tailDict[r].keys())
                relations.add(r)
            for r in self.headDict:
                hpt[r] = 0
                for t in self.headDict[r]:
                    hpt[r] += len(self.headDict[r][t])
                hpt[r] = hpt[r]/len(self.headDict[r].keys())
                relations.add(r)
            self.headProb = {}
            for r in relations:
                self.headProb[r] = tph[r]/(tph[r]+hpt[r])
                #self.tailProb[r] = hpt[r]/(tph[r]+hpt[r])

        if seed is not None:
            random.seed(seed)

        # The entity set and anomalies must be the same for all
        self.entitySet = set(range(loaders[0].entityTotal))
        self.relation_anomaly = loaders[0].relation_anomaly
        self.tripleTotal = len(self.tripleList)
        if self.nbatches is not None:
            self.batchSize = self.tripleTotal // self.nbatches

        self.headCorruptedDict = {}
        self.tailCorruptedDict = {}
        self.corruption_mode = corruption_mode

        self.headEntities, self.tailEntities = {}, {}
        self.headEntities[-1] = list(self.entitySet - headEntities)
        self.tailEntities[-1] = list(self.entitySet - tailEntities)

        self.headCorruptedEntities = 0
        self.tailCorruptedEntities = 0

        # Nothing to do when using global
        if self.corruption_mode != "Global":
            for r in self.headDict:
                self.headCorruptedDict[r] = {}

                headEntities = set()
                if self.corruption_mode == "LCWA":
                    headEntities = self.entitySet
                elif self.corruption_mode == "TCLCWA":
                    headEntities = dom[r]
                elif self.corruption_mode == "NLCWA" or self.corruption_mode == "GNLCWA":
                    headEntities = dom[r]
                    # Compatible relations are always the same.
                    for ri in loaders[0].domDomCompatible[r]:
                        headEntities = headEntities.union(ran[ri])
                        #headEntities = headEntities.union(dom[ri])
                    for rj in loaders[0].domRanCompatible[r]:
                        headEntities = headEntities.union(dom[rj])
                        #headEntities = headEntities.union(ran[rj])
                self.headEntities[r] = list(headEntities)

                for t in self.headDict[r]:
                    corruptedHeads = headEntities - self.headDict[r][t]
                    if len(corruptedHeads) == 0 and self.corruption_mode == "LCWA":
                        print("Corrupted heads were empty using LCWA")
                        sys.exit(-1)
                    # Only add the key if there are available entities.
                    if len(corruptedHeads) != 0:
                        self.headCorruptedDict[r][t] = 0

            for r in self.tailDict:
                self.tailCorruptedDict[r] = {}

                tailEntities = set()
                if self.corruption_mode == "LCWA":
                    tailEntities = self.entitySet
                elif self.corruption_mode == "TCLCWA":
                    tailEntities = ran[r]
                elif self.corruption_mode == "NLCWA" or self.corruption_mode == "GNLCWA":
                    tailEntities = ran[r]
                    for ri in loaders[0].ranRanCompatible[r]:
                        tailEntities = tailEntities.union(dom[ri])
                        #tailEntities = tailEntities.union(ran[ri])
                    for rj in loaders[0].ranDomCompatible[r]:
                        tailEntities = tailEntities.union(ran[rj])
                        #tailEntities = tailEntities.union(dom[rj])
                self.tailEntities[r] = list(tailEntities)

                for h in self.tailDict[r]:
                    corruptedTails = tailEntities - self.tailDict[r][h]
                    if len(corruptedTails) == 0 and self.corruption_mode == "LCWA":
                        print("Corrupted tails were empty using LCWA")
                        sys.exit(-1)
                    # Only add the key if there are available entities.
                    if len(corruptedTails) != 0:
                        self.tailCorruptedDict[r][h] = 0

    def corrupt_head(self, h, r, t):
        useGlobal = random.random() < 0.25
        if self.corruption_mode == "Global" or (self.corruption_mode == "GNLCWA" and useGlobal):
            hPrime = self.headEntities[-1][self.headCorruptedEntities]
            self.headCorruptedEntities = self.headCorruptedEntities + 1
            if self.headCorruptedEntities == len(self.headEntities[-1]):
                self.headCorruptedEntities = 0
        else:
            hPrime = self.next_corrupted(t, self.headCorruptedDict[r], self.headEntities[r], self.headDict[r])
        return hPrime

    def corrupt_tail(self, h, r, t):
        useGlobal = random.random() < 0.25
        if self.corruption_mode == "Global" or (self.corruption_mode == "GNLCWA" and useGlobal):
            tPrime = self.tailEntities[-1][self.tailCorruptedEntities]
            self.tailCorruptedEntities = self.tailCorruptedEntities + 1
            if self.tailCorruptedEntities == len(self.tailEntities[-1]):
                self.tailCorruptedEntities = 0
        else:
            tPrime = self.next_corrupted(h, self.tailCorruptedDict[r], self.tailEntities[r], self.tailDict[r])
        return tPrime

    def next_corrupted(self, e, corrupted_dict, all_entities, these_entities):
        ret = None
        if e in corrupted_dict:
            while True:
                ret = all_entities[corrupted_dict[e]]
                corrupted_dict[e] = corrupted_dict[e] + 1
                if corrupted_dict[e] == len(all_entities):
                    corrupted_dict[e] = 0
                if ret not in these_entities[e]:
                    return ret

    """ Corrupted heads or tails. """
    def get_corrupted(self, h, r, t, type='head'):
        if self.corruption_mode == "Global":
            if type == "head":
                corrupted = self.headEntities[-1]
            elif type == "tail":
                corrupted = self.tailEntities[-1]
        else:
            if type == "head":
                corrupted = set(self.headEntities[r]) - self.headDict[r][t]
            elif type == "tail":
                corrupted = set(self.tailEntities[r]) - self.tailDict[r][h]
        return corrupted

    def get_triples(self):
        return self.tripleList

    def getBatches(self):
        batch_seq_size = self.batchSize * (1 + self.negative_ent + self.negative_rel)
        batch_h = np.zeros(batch_seq_size, dtype=np.int64)
        batch_t = np.zeros(batch_seq_size, dtype=np.int64)
        batch_r = np.zeros(batch_seq_size, dtype=np.int64)
        batch_y = np.zeros(batch_seq_size, dtype=np.float32)

        for batch in range(self.batchSize):
            randIndex = random.randint(0, self.tripleTotal-1)
            batch_h[batch] = self.tripleList[randIndex].h
            batch_t[batch] = self.tripleList[randIndex].t
            batch_r[batch] = self.tripleList[randIndex].r
            batch_y[batch] = 1
            last = self.batchSize

            for times in range(self.negative_ent):
                ch = self.tripleList[randIndex].h
                ct = self.tripleList[randIndex].t

                if random.random() < self.headProb[self.tripleList[randIndex].r] \
                    if self.use_bern else random.random() < 0.5:
                    ch = self.corrupt_head(ch, self.tripleList[randIndex].r, ct)
                else:
                    ct = self.corrupt_tail(ch, self.tripleList[randIndex].r, ct)

                if ch == None or ct == None:
                    times = times - 1
                    continue

                batch_h[batch + last] = ch
                batch_t[batch + last] = ct
                batch_r[batch + last] = self.tripleList[randIndex].r
                batch_y[batch + last] = -1
                last = last + self.batchSize

        return {
            "batch_h": batch_h,
            "batch_t": batch_t,
            "batch_r": batch_r,
            "batch_y": batch_y,
            "mode": "normal"
        }

    def __next__(self):
        self.counter += 1
        if self.counter > self.nbatches:
            raise StopIteration()
        return self.getBatches()

    def __iter__(self):
        return DataSampler(self.nbatches, self.getBatches)

    def __len__(self):
        return self.nbatches