import torch
from Train.train import train
from Train.test import test
model_name = "transe"

#Datasets numbered from 0-7
dataset = 6

corruption = "Global"

parameters = {}
parameters["nbatches"] = 225
parameters["nr"] = 13
parameters["lr"] = 0.7801073119264197
parameters["wd"] = 1.4175317295064768e-09
parameters["m"] = 0.9263060805387795
parameters["trial_index"] = 1
parameters["dim"] = 40
#parameters["dimr"] = 30
parameters["pnorm"] = 2
parameters["norm"] = True
parameters["gamma"] = 4.226494273198768

train(model_name = model_name, dataset = dataset, corruption_mode = corruption, parameters = parameters, inner_norm=True)

#test(model_name = model_name, dataset = dataset, corruption_mode = corruption)



