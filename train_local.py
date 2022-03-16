from DataLoader.TripleManager import TripleManager
from Train.Evaluator import Evaluator
from Train.Trainer import Trainer
from Utils import ModelUtils
from Utils import LossUtils
import torch
import time
import os


def get_params(index, total_points):
    current = 0
    for model_name in ['transe', 'transh', 'transd', 'distmult', 'complex', 'hole', 'simple', 'rotate']:
        for dataset in range(0, 8):
            for point in range(0, total_points):
                # This is the prefix to point to different splits. We have:
                #   (empty): Original split
                if current == index:
                    return model_name, dataset, '', point
                current += 1

def run():
    folder = ''
    model_name, dataset, split_prefix, point = 'hole', 6, '', 0

    rel_anomaly_min = 0
    rel_anomaly_max = 1.0

    validation_epochs = 10
    train_times = 500

    use_gpu = True

    corruption_mode = "LCWA"

    parameters = {}
    parameters["batch_size"] = 1000
    parameters["nr"] = 100
    parameters["trial_index"] = 1
    parameters["inner_norm"] = False
    parameters["norm"] = True
    parameters["dim"] = 5
    parameters["dime"] = 5
    parameters["dimr"] = 5
    parameters["lr"] = 0.6164507232989961
    parameters["wd"] = 0.011878624460305654
    parameters["m"] = 0.511354367248714
    parameters["gamma"] = 2
    parameters["pnorm"] = 2

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

    print("Model:", model_name, "; Dataset:", dataset_name, "; Corruption:", corruption_mode)

    start = time.perf_counter()
    path = folder + "Datasets/" + dataset_name + "/"
    train_manager = TripleManager(path, splits=[split_prefix+"train"], batch_size=parameters["batch_size"],
                                  neg_rate=parameters["nr"],  corruption_mode=corruption_mode)
    parameters["ent_total"] = train_manager.entityTotal
    parameters["rel_total"] = train_manager.relationTotal
    print("Parameters:", parameters)
    
    mu = ModelUtils.getModel(model_name, parameters)
    mu.set_params(parameters)
    print("Model name : ", mu.model_name)
    loss = LossUtils.getLoss(gamma = parameters["gamma"], model = mu)
    
    validation = Evaluator(TripleManager(path, splits=[split_prefix+"valid"],
                batch_size=parameters["batch_size"], neg_rate=parameters["nr"], corruption_mode=corruption_mode),
                           rel_anomaly_max=rel_anomaly_max, rel_anomaly_min=rel_anomaly_min, use_gpu=use_gpu)
    
    end = time.perf_counter()
    print("Initialization time: " + str(end - start))

    start = time.perf_counter()
    checkpoint_dir=folder + "Model/" + str(dataset) + "/" + model_name + "_" + split_prefix + "_" + str(point)

    init_epoch = 0
    # There is a checkpoint, let's load it!
    if os.path.exists(checkpoint_dir + ".ckpt"):
        loss.model = torch.load(checkpoint_dir + ".ckpt")
        init_epoch = loss.model.epoch

    # If this exists, we are done; otherwise, let's go for it.
    if not os.path.exists(checkpoint_dir + ".model"):
        trainer = Trainer(model=loss, train=train_manager, validation=validation, train_times=train_times,
                    alpha=parameters["lr"], weight_decay=parameters["wd"], momentum=parameters["m"], use_gpu=use_gpu,
                    save_steps=validation_epochs, checkpoint_dir=checkpoint_dir, inner_norm=parameters["inner_norm"],
                           opt_method='adam')
        trainer.run(init_epoch=init_epoch)
    else:
        print("Model already exists")
    end = time.perf_counter()
    print("Time elapsed during the calculation: " + str(end - start))

    # We are done! Rename checkpoint to model.
    if not os.path.exists(checkpoint_dir + ".model"):
        os.rename(checkpoint_dir + ".valid", checkpoint_dir + ".model")
        os.remove(checkpoint_dir + ".ckpt")

if __name__ == '__main__':
    run()