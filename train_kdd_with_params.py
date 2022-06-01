from DataLoader.TripleManager import TripleManager
from Train.Evaluator import Evaluator
from Train.Trainer import Trainer
from Utils import ModelUtils
from Utils import LossUtils
import torch
import time
import os
import sys
import torch.optim as optim


def get_params(index):
    current = 0
    for model_name in ['transe', 'transh', 'transd', 'transr', 'distmult', 'complex', 'hole', 'simple', 'rotate',
                       'analogy', 'quate', 'toruse']:
        for dataset in range(0, 8):
            # This is the prefix to point to different splits. We have:
            #   (empty): Original split
            #   new_: The new split we did for WWW'21 using Kolmogorov-Smirnov
            #   split_ks_zerofive_: Using KGSplit based on Kolmogorov-Smirnov
            #   split_c_zerofive_: Using KGSplit based on Cucconi
            for split_prefix in ['', 'split_ks_zerofive_', 'split_c_zerofive_']:
                if current == index:
                    return model_name, dataset, split_prefix
                current += 1


def run():
    folder = sys.argv[1]
    index = int(sys.argv[2])

    model_name, dataset, split_prefix = get_params(index)

    # The Evaluator throws a memory error for TransD (all datasets) and TransR, YAGO3-10.
    validation_batched = model_name is 'transd' or (model_name is 'transr' and dataset is 7)

    rel_anomaly_min = 0
    rel_anomaly_max = 1.0

    validation_epochs = 100
    train_times = 5000

    use_gpu = False

    corruption_mode = "LCWA"

    parameters = {}
    # Batch size
    parameters["batch_size"] = 1000

    # Negative rate
    parameters["nr"] = 5

    # Embedding dimensions
    parameters["dim"] = 75
    parameters["dime"] = 75
    parameters["dimr"] = 75

    # Optimizer
    parameters["lr"] = None
    parameters["weight_decay"] = None
    parameters["momentum"] = None
    parameters["opt_method"] = "adam"

    # Regularization
    parameters["lmbda"] = 1e-5
    parameters["reg_type"] = 'L2'

    # Margin
    parameters["gamma"] = 1e-4

    # Norm used in score
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

    print("Model:", model_name, "; Dataset:", dataset_name, "; Index:", index,
          "; Corruption:", corruption_mode, '; Parameters: ', parameters)

    start = time.perf_counter()
    path = folder + "Datasets/" + dataset_name + "/"
    train_manager = TripleManager(path, splits=[split_prefix + "train"], batch_size=parameters["batch_size"],
                                  neg_rate=parameters["nr"], corruption_mode=corruption_mode)
    parameters["ent_total"] = train_manager.entityTotal
    parameters["rel_total"] = train_manager.relationTotal
    # TODO Add extra parameters here!

    mu = ModelUtils.getModel(model_name, parameters)
    mu.set_params(parameters)
    print("Model name : ", mu.get_model_name())

    loss = LossUtils.getLoss(gamma=parameters["gamma"], model=mu, reg_type=parameters["reg_type"])

    validation = Evaluator(TripleManager(path, splits=[split_prefix + "valid", split_prefix + "train"],
                                         batch_size=parameters["batch_size"], neg_rate=parameters["nr"],
                                         corruption_mode=corruption_mode),
                           rel_anomaly_max=rel_anomaly_max, rel_anomaly_min=rel_anomaly_min, batched=validation_batched)

    checkpoint_dir = folder + "Model/" + str(dataset) + "/" + model_name + "_" + split_prefix + "_" + str(index)

    init_epoch = 0
    # There is a checkpoint, let's load it!
    if os.path.exists(checkpoint_dir + ".ckpt"):
        loss.model = torch.load(checkpoint_dir + ".ckpt")
        init_epoch = loss.model.epoch
    else:
        # Initialize model from scratch
        loss.model.initialize_model()
    mu.set_use_gpu(use_gpu)

    end = time.perf_counter()
    print("Initialization time: " + str(end - start))

    start = time.perf_counter()

    # load valid function.
    def load_valid():
        valid = None
        # There is a validation model, let's load it!
        if os.path.exists(checkpoint_dir + ".valid"):
            valid = torch.load(checkpoint_dir + ".valid")
        return valid

    # Save valid function.
    def save_valid():
        torch.save(loss.model, os.path.join(checkpoint_dir + ".valid"))

    # Save the current checkpoint.
    def save_checkpoint():
        torch.save(loss.model, os.path.join(checkpoint_dir + ".ckpt"))

    # https://stackoverflow.com/questions/52494128/call-function-without-optional-arguments-if-they-are-none
    # Note that lr is mandatory for SGD and the others do not have momentum.
    optimargs = {k: v for k, v in dict(
        lr=parameters["lr"],
        weight_decay=parameters["weight_decay"],
        momentum=parameters["momentum"], ).items() if v is not None}

    optimizer = None
    if parameters["opt_method"] == "adam":
        optimizer = optim.Adam(
            loss.parameters(),
            **optimargs,
        )

    # If this exists, we are done; otherwise, let's go for it.
    if not os.path.exists(checkpoint_dir + ".model"):
        trainer = Trainer(loss=loss, train=train_manager, validation=validation, train_times=train_times,
                          save_steps=validation_epochs, checkpoint_dir=checkpoint_dir, load_valid=load_valid,
                          save_valid=save_valid, save_checkpoint=save_checkpoint, optimizer=optimizer, )
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
