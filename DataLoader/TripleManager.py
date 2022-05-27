import numpy as np
import random
import sys
import math
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
    """
    pairing_mode is either Paired or Unpaired. In Paired, each positive triple must be paired with negatives. This 
    restriction does not exist for Unpaired. 
    """
    # TODO Change splits for a dictionary?
    def __init__(self, path, splits, batch_size = None, neg_rate = None, use_bern=False, seed=None,
                 corruption_mode="Global", pairing_mode="Paired"):
        self.counter = 0
        # Whether we will use a Bernoulli distribution to determine whether to corrupt head or tail
        
        self.use_bern = use_bern
        loaders = []
        self.tripleList = []
        headEntities, tailEntities = set(), set()
        self.headDict, self.tailDict, dom, ran = {}, {}, {}, {}
        self.triple_count_by_pred = {}

        for s in splits:
            loader = DataLoader(path, s)
            self.entityTotal = loader.entityTotal
            self.relationTotal = loader.relationTotal

            for r in loader.triple_count_by_pred:
                if r not in self.triple_count_by_pred:
                    self.triple_count_by_pred[r] = {}
                    self.triple_count_by_pred[r]['global'] = 0
                self.triple_count_by_pred[r]['global'] += loader.triple_count_by_pred[r]

            loaders = loaders + [loader]
            # The triples are only the ones in the first loader. The other loaders are just for filtering existing
            #   triples in other splits.
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

        # This is domain/range
        self.triple_count_by_pred_loc = {}
        for r in self.headDict:
            self.triple_count_by_pred_loc[r] = {}
            self.triple_count_by_pred_loc[r]['domain'] = len(self.headDict[r])
        for r in self.tailDict:
            if r not in self.triple_count_by_pred_loc:
                self.triple_count_by_pred_loc[r] = {}
            self.triple_count_by_pred_loc[r]['range'] = len(self.tailDict[r])

        self.allHeadEntities = headEntities
        self.allTailEntities = tailEntities
       
        if self.use_bern:
            # tph: the average number of tail entities per head entity
            # hpt: the average number of head entities per tail entity
            tph, hpt = {}, {}
            relations = set()
            for r in self.tailDict:
                tph[r] = 0
                for h in self.tailDict[r]:
                    tph[r] += len(self.tailDict[r][h])
                tph[r] = tph[r] / len(self.tailDict[r].keys())
                relations.add(r)
            for r in self.headDict:
                hpt[r] = 0
                for t in self.headDict[r]:
                    hpt[r] += len(self.headDict[r][t])
                hpt[r] = hpt[r] / len(self.headDict[r].keys())
                relations.add(r)
            self.headProb = {}
            for r in relations:
                self.headProb[r] = tph[r] / (tph[r] + hpt[r])
                # self.tailProb[r] = hpt[r]/(tph[r]+hpt[r])

        if seed is not None:
            # TODO Are these two seeds necessary?
            np.random.seed(seed)
            random.seed(seed)

        # The entity set and anomalies must be the same for all
        self.entitySet = set(range(loaders[0].entityTotal))
        self.relSet = set(range(loaders[0].relationTotal))
        self.relation_anomaly = loaders[0].relation_anomaly
        self.tripleTotal = len(self.tripleList)

        self.batch_size = batch_size
        self.neg_rate = neg_rate

        if self.batch_size != None:
            self.nbatches = math.ceil(len(self.tripleList)/self.batch_size)
        self.randIndexes = np.array([i for i in range(0, len(self.tripleList))])
        np.random.shuffle(self.randIndexes)
        self.headCorruptedDict = {}
        self.tailCorruptedDict = {}
        self.corruption_mode = corruption_mode
        self.pairing_mode = pairing_mode

        self.headEntities, self.tailEntities = {}, {}
        self.headEntities[-1] = list(self.entitySet - headEntities)
        self.tailEntities[-1] = list(self.entitySet - tailEntities)

        if len(self.headEntities[-1]) == 0 and self.corruption_mode == "Global":
            print("Heads were empty using Global")
            sys.exit(-1)
        if len(self.tailEntities[-1]) == 0 and self.corruption_mode == "Global":
            print("Tails were empty using Global")
            sys.exit(-1)

        self.headCorruptedEntities = 0
        self.tailCorruptedEntities = 0

        if self.corruption_mode == "Global":
            # This is to make Global work as the rest.
            for r in self.relSet:
                # First, the head entities are those that are tails only.
                self.headEntities[r] = self.tailEntities[-1]
                self.tailEntities[r] = self.headEntities[-1]
            for r in self.headDict:
                self.headCorruptedDict[r] = {}
                for t in self.headDict[r]:
                    self.headCorruptedDict[r][t] = 0
            for r in self.tailDict:
                self.tailCorruptedDict[r] = {}
                for h in self.tailDict[r]:
                    self.tailCorruptedDict[r][h] = 0
        else:
            for r in self.headDict:
                self.headCorruptedDict[r] = {}

                headEntities = set()
                if self.corruption_mode == "LCWA":
                    headEntities = self.entitySet
                elif self.corruption_mode == "TCLCWA":
                    headEntities = dom[r]
                elif self.corruption_mode == "NLCWA":
                    headEntities = dom[r]
                    # Compatible relations are always the same.
                    for ri in loaders[0].domDomCompatible[r]:
                        headEntities = headEntities.union(ran[ri])
                        # headEntities = headEntities.union(dom[ri])
                    for rj in loaders[0].domRanCompatible[r]:
                        headEntities = headEntities.union(dom[rj])
                        # headEntities = headEntities.union(ran[rj])
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
                elif self.corruption_mode == "NLCWA":
                    tailEntities = ran[r]
                    for ri in loaders[0].ranRanCompatible[r]:
                        tailEntities = tailEntities.union(dom[ri])
                        # tailEntities = tailEntities.union(ran[ri])
                    for rj in loaders[0].ranDomCompatible[r]:
                        tailEntities = tailEntities.union(ran[rj])
                        # tailEntities = tailEntities.union(dom[rj])
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
        # TODO This should also work for Global now.
        return self.next_corrupted(t, self.headCorruptedDict[r], self.headEntities[r], self.headDict[r])

    def corrupt_tail(self, h, r, t):
        # TODO This should also work for Global now.
        return self.next_corrupted(h, self.tailCorruptedDict[r], self.tailEntities[r], self.tailDict[r])

    def next_corrupted(self, e, corrupted_dict, all_entities, these_entities):
        if e in corrupted_dict:
            while True:
                ret = all_entities[corrupted_dict[e]]
                corrupted_dict[e] = corrupted_dict[e] + 1
                if corrupted_dict[e] == len(all_entities):
                    corrupted_dict[e] = 0
                if ret not in these_entities[e]:
                    return ret

    """ All corrupted heads or tails. """
    def get_corrupted(self, h, r, t, type='head'):
        # TODO This should work now without making any distinction: headEntities and tailEntities should point to -1
        #   for every relation when using Global.
        if type == "head":
            corrupted = set(self.headEntities[r]) - self.headDict[r][t]
        elif type == "tail":
            corrupted = set(self.tailEntities[r]) - self.tailDict[r][h]
        return corrupted

    def get_triples(self):
        return self.tripleList

    def get_batch(self):
        # We will have batch_size positives and batch_size*neg_rate negatives.

        # Instead of picking a triple by random, I have created a list with all triple indexes, and shuffled them
        # For example, if we have 5 triples, randIndexes = [3,2,1,4,0]
        # So now what we are doing is we are picking the first batch_size triple indexes from randIndexes and thats how we choose a positive triple
        # At each iteration, we are picking the next batch_size elements, until we no longer have enough elements
        # In that case, our last batch contains the last remaining triples

        bs = self.batch_size if self.batch_size<=len(self.randIndexes) else len(self.randIndexes)
        
        batch_seq_size = bs * (1 + self.neg_rate)
        batch_h = np.zeros(batch_seq_size, dtype=np.int64)
        batch_t = np.zeros(batch_seq_size, dtype=np.int64)
        batch_r = np.zeros(batch_seq_size, dtype=np.int64)
        batch_y = np.zeros(batch_seq_size, dtype=np.float32)

        for i_in_batch in range(bs):
            # Get random positive triple.
            
            batch_h[i_in_batch] = self.tripleList[self.randIndexes[i_in_batch]].h
            batch_t[i_in_batch] = self.tripleList[self.randIndexes[i_in_batch]].t
            batch_r[i_in_batch] = self.tripleList[self.randIndexes[i_in_batch]].r
            batch_y[i_in_batch] = 1
            last = bs

            for times in range(self.neg_rate):
                # The corrupted head and tail. We will not corrupt the relation.
                ch = self.tripleList[self.randIndexes[i_in_batch]].h
                ct = self.tripleList[self.randIndexes[i_in_batch]].t
                r = self.tripleList[self.randIndexes[i_in_batch]].r

                # If it is paired, it will corrupt either head or tail. If unpaired, it will corrupt head and tail
                #   several times (random number between 1 and 10).
                for corruptions in range(1 if self.pairing_mode == 'Paired' else random.randint(1, 10)):
                    if random.random() < self.headProb[self.tripleList[self.randIndexes[i_in_batch]].r] \
                            if self.use_bern else random.random() < 0.5:
                        ch = self.corrupt_head(ch, r, ct)
                    else:
                        
                        ct = self.corrupt_tail(ch, r, ct)

                # TODO Do we need this?
                #if ch == None or ct == None:
                #    times = times - 1
                #    continue

                batch_h[i_in_batch + last] = ch
                batch_t[i_in_batch + last] = ct
                batch_r[i_in_batch + last] = self.tripleList[self.randIndexes[i_in_batch]].r
                batch_y[i_in_batch + last] = -1
                last = last + bs
        self.randIndexes = self.randIndexes[bs:]
        return {
            "batch_h": batch_h,
            "batch_t": batch_t,
            "batch_r": batch_r,
            "batch_y": batch_y
        }



    def __next__(self):
        # This is required by python to iterate through an object
        self.counter += 1
        if self.counter > self.nbatches:
            raise StopIteration()

        return self.get_batch()

    def __iter__(self):
        # Here at the beginning of every epoch, I am setting counter to 0
        # And resetting randIndexes at the beginning of every epoch
        self.counter = 0
        self.randIndexes = np.array([i for i in range(0, len(self.tripleList))])
        np.random.shuffle(self.randIndexes)
        return self

    def __len__(self):
        return self.nbatches