from Train.train import train
from Utils.Embedding import Embedding
from Loss.MarginLoss import MarginLoss
from Loss.SigmoidLoss import SigmoidLoss
import torch
from DataLoader.TripleManager import TripleManager


path = "Datasets/" + "WN18RR" + "/"
# train = TripleManager(path, splits=["new_train"], nbatches=10,
#                                   neg_ent=10, neg_rel=6, corruption_mode="Global")
train = TripleManager(path, splits=["new_train"], batch_size = 8912,
                                    neg_rate=10, corruption_mode="Global")
dictOfActualTriples = {}

f = open("Datasets/WN18RR/new_train2id.txt")
l = []
ctr = 0
for line in f:
    if ctr ==0:
        ctr+=1
        continue
    splits = line.strip().split(" ")
    l.append((int(splits[0]), int(splits[2]), int(splits[1])))

setOfAllTriples = set(l)
actualTotalTriples = 0
for data in train:
    actualTotalTriples+= len(data["batch_h"])
    for i in range(0, len(data["batch_h"])):
        if data["batch_y"][i] == 1:
            triple = (int(data["batch_h"][i]), int(data["batch_r"][i]), int(data["batch_t"][i]))

            if triple in dictOfActualTriples:
                dictOfActualTriples[triple] +=1
            else:
                dictOfActualTriples[triple] = 1

print ("Expected number of total triples in train (88912 triples in test * (1+nr)): ", 978032)
print ("Actual number of total triples in train : ", actualTotalTriples)
print ("Expected number of unique triples in train : ", len(setOfAllTriples))
print ("Actual number of unique triples in train : ", len(dictOfActualTriples))

print ("Missing triples : ")
for triple in setOfAllTriples:
    if triple not in dictOfActualTriples:
        print (triple)


#print (train.getBatches())
# train2 = TripleManager_new(path, splits=["new_train"], batch_size = 8912,
#                                    neg_rate=10, corruption_mode="LCWA")