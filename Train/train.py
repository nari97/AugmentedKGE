from DataLoader.TripleManager import TripleManager
from Train.Evaluator import Evaluator
from Train.Trainer import Trainer
from Models.ModelUtils import ModelUtils
from Train.Evaluator import RankCollector
import numpy as np
import time
import sys
import os
import pickle

def train():
    #folder = sys.argv[1]
    #model_name = sys.argv[2]
    #dataset = int(sys.argv[3])
    #index = int(sys.argv[4])
    #corruption_mode = "LCWA"

    folder = ""
    model_name = "complex"
    dataset = 6
    index = 0
    corruption_mode = "Global"

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
    
    
    mu = ModelUtils(model_name, parameters)
    print("Parameters:", parameters)
    
    
    
    start = time.perf_counter()
    path = folder + "Datasets/" + dataset_name + "/"
    train_manager = TripleManager(path, splits=["new_train"], nbatches=parameters["nbatches"],
                                  neg_ent=parameters["nr"], neg_rel=negative_rel, corruption_mode=corruption_mode)

    model = mu.get_model(train_manager.entityTotal, train_manager.relationTotal, train_manager.batchSize)

    validation = Evaluator(TripleManager(path, splits=["new_valid", "new_train"], corruption_mode=corruption_mode),
                           rel_anomaly_max=rel_anomaly_max, rel_anomaly_min=rel_anomaly_min)
    end = time.perf_counter()
    print("Initialization time: " + str(end - start))

    start = time.perf_counter()
    checkpoint_dir=folder + "Model/" + str(dataset) + "/" + mu.get_name()
    init_epoch = 0
    if os.path.exists(checkpoint_dir + ".ckpt"):
        model.model.load_checkpoint(checkpoint_dir + ".ckpt")
        init_epoch = model.model.epoch
    if not os.path.exists(checkpoint_dir + ".model"):

        trainer = Trainer(model=model, train=train_manager, validation=validation, train_times=train_times,
            alpha=parameters["lr"], weight_decay=parameters["wd"], momentum=parameters["m"], use_gpu=False,
                          save_steps=validation_epochs, checkpoint_dir=checkpoint_dir)
        
        trainer.run(init_epoch=init_epoch)
    else:
        print("Model already exists")
    end = time.perf_counter()
    print("Time elapsed during the calculation: " + str(end - start))

    # We are done! Rename checkpoint to model.
    if not os.path.exists(checkpoint_dir + ".model"):
        os.rename(checkpoint_dir + ".valid", checkpoint_dir + ".model")
        os.remove(checkpoint_dir + ".ckpt")

    # Report metric
    with open(checkpoint_dir + ".ranks", 'rb') as f:
        all_ranks = np.load(f)
    with open(checkpoint_dir + ".totals", 'rb') as f:
        all_totals = np.load(f)
    rc = RankCollector()
    rc.load(all_ranks.tolist(), all_totals.tolist())

    # Report metric!!!!!
    result = {}
    result['trial_index'] = parameters['trial_index']
    result['mrh'] = rc.get_metric().get()
    result_file = folder + "Ax/" + model_name + "_" + str(dataset) + "_" + str(index) + ".result"
    with open(result_file, 'wb') as f:
        pickle.dump(result, f, pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    train()