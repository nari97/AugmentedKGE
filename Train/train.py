from DataLoader.TripleManager import TripleManager
from Train.Evaluator import Evaluator
from Train.Trainer import Trainer
from Train.Evaluator import RankCollector
from Strategy.NegativeSampling import NegativeSampling
from Utils import utils
import torch
import Loss
import numpy as np
import time
import sys
import os
import pickle

def train(model_name, dataset, corruption_mode, parameters, index = 0, validation_epochs = 10, train_times = 2500, rel_anomaly_min = 0, rel_anomaly_max = 0.75):
    #folder = sys.argv[1]
    #model_name = sys.argv[2]
    #dataset = int(sys.argv[3])
    #index = int(sys.argv[4])
    #corruption_mode = "LCWA"

    folder = ""
    

    validation_epochs = 10
    train_times = 2500
    negative_rel = 0
    rel_anomaly_max = .75
    rel_anomaly_min = 0

    dataset_name = ""
    if dataset == 0:
        dataset_name = "FB13"
    if dataset == 1:
        dataset_name = "FB15K"
    if dataset == 2:
        dataset_name = "FB15K237"
    if dataset == 3:
        dataset_name = "NELL-995"
    if dataset == 4:
        dataset_name = "WN11"
    if dataset == 5:
        dataset_name = "WN18"
    if dataset == 6:
        dataset_name = "WN18RR"
    if dataset == 7:
        dataset_name = "YAGO3-10"

    print("Model:", model_name, "; Dataset:", dataset_name, "; Index:", index, "; Corruption:", corruption_mode)
    
    '''
    trial_file = folder + "Ax/" + model_name + "_" + str(dataset) + "_" + str(index) + ".trial"
    if not os.path.exists(trial_file):
        return

    # Get parameters and index for trial
    
    with open(trial_file, 'rb') as f:
        parameters = pickle.load(f)
    '''
    
    #utils.check_params(parameters)

    start = time.perf_counter()
    path = folder + "Datasets/" + dataset_name + "/"
    train_manager = TripleManager(path, splits=["new_train"], nbatches=parameters["nbatches"],
                                  neg_ent=parameters["nr"], neg_rel=negative_rel, corruption_mode=corruption_mode)
    parameters["ent_total"] = train_manager.entityTotal
    parameters["rel_total"] = train_manager.relationTotal
    print("Parameters:", parameters)


    mu = utils.getModel(model_name, parameters)
    mu.set_params(parameters)
    
    loss = utils.getLoss(model_name, parameters["gamma"])

    model = NegativeSampling(model = mu, loss = loss, batch_size = train_manager.batchSize)
    for name, param in mu.named_parameters():
        print (name)
    exit()
    validation = Evaluator(TripleManager(path, splits=["new_valid", "new_train"], corruption_mode=corruption_mode),
                           rel_anomaly_max=rel_anomaly_max, rel_anomaly_min=rel_anomaly_min)

    end = time.perf_counter()
    print("Initialization time: " + str(end - start))

    start = time.perf_counter()
    # checkpoint_dir=folder + "Model/" + str(dataset) + "/" + mu.get_name()
    checkpoint_dir=folder + "Model/" + model_name
    init_epoch = 0
    if os.path.exists(checkpoint_dir + ".ckpt"):
        #model.model.load_checkpoint(checkpoint_dir + ".ckpt")
        model.model = torch.load(checkpoint_dir + ".ckpt")
        init_epoch = model.model.epoch
    if not os.path.exists(checkpoint_dir + ".model"):

        trainer = Trainer(model=model, train=train_manager, validation=validation, train_times=train_times,
            alpha=parameters["lr"], weight_decay=parameters["wd"], momentum=parameters["m"], use_gpu=False,
                          save_steps=validation_epochs, checkpoint_dir=checkpoint_dir, inner_norm = parameters["inner_norm"])
        
        trainer.run(init_epoch=init_epoch)
    else:
        print("Model already exists")
    end = time.perf_counter()
    print("Time elapsed during the calculation: " + str(end - start))

    # We are done! Rename checkpoint to model.
    if not os.path.exists(checkpoint_dir + ".model"):
        os.rename(checkpoint_dir + ".valid", checkpoint_dir + ".model")
        os.remove(checkpoint_dir + ".ckpt")

    # Report metrics
    utils.reportMetrics(folder, dataset, index, model_name, model.model.ranks, model.model.totals, parameters)





