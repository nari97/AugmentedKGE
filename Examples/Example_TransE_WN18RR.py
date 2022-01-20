import torch
from Train.train import train
from Train.test import test
model_name = "transe"
dataset = 6
corruption = "Global"
parameters = {}
parameters["batch_size"] = 10000
parameters["nr"] = 13
parameters["lr"] = 0.37621418858596917
parameters["wd"] = 1.6566904670936488e-07
parameters["m"] = 0.9890299829607658
parameters["trial_index"] = 1
parameters["dim"] = 174
parameters["dimr"] = 30
parameters["dime"] = 40
parameters["pnorm"] = 1
parameters["gamma"] = 7.746041492315607
parameters["bern"] = True
parameters["inner_norm"] = True

train(model_name = model_name, dataset = dataset, corruption_mode = corruption, parameters = parameters, use_gpu = True)
test(model_name = model_name, dataset = dataset, corruption_mode = corruption, use_gpu = True)

