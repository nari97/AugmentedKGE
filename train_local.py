from DataLoader.TripleManager import TripleManager
from Train.Evaluator import Evaluator
from Train.Trainer import Trainer
from Utils import ModelUtils
from Utils import LossUtils
import torch
import torch.optim as optim
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
    model_name, dataset, split_prefix, point = 'makr', 6, '', 0

    rel_anomaly_min = 0
    rel_anomaly_max = 1.0

    validation_epochs = 5
    train_times = 500

    use_gpu = False

    corruption_mode = "Global"

    parameters = {}
    parameters["batch_size"] = 1000
    parameters["nr"] = 5
    parameters["trial_index"] = 1
    parameters["dim"] = 5
    parameters["dime"] = 5
    parameters["dimr"] = 3

    parameters["lr"] = None

    # TODO Testing regularization
    parameters["lmbda"] = 1e-5
    parameters["reg_type"] = 'L2'

    # Weight of constraints over parameters.
    parameters["weight_constraints"] = 1e-5
    # TODO Normalization of lambda: https://arxiv.org/pdf/1711.05101.pdf (Appendix B.1)

    # TODO This is L2 regularization!
    #   https://towardsdatascience.com/weight-decay-and-its-peculiar-effects-66e0aee3e7b8
    #   See also: https://github.com/zeke-xie/stable-weight-decay-regularization
    # If you set weight decay, you are using L2 regularization without control.
    parameters["weight_decay"] = None
    parameters["momentum"] = None
    parameters["opt_method"] = "adam"

    parameters["gamma"] = 1e-4
    parameters["other_gamma"] = 1e-4

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

    # Only for TransSparse; either share or separate
    parameters["sparse_type"] = 'share'
    print("Parameters:", parameters)

    start = time.perf_counter()
    path = folder + "Datasets/" + dataset_name + "/"
    train_manager = TripleManager(path, splits=[split_prefix + "train"], batch_size=parameters["batch_size"],
                                  neg_rate=parameters["nr"], corruption_mode=corruption_mode)
    parameters["ent_total"] = train_manager.entityTotal
    parameters["rel_total"] = train_manager.relationTotal
    parameters["pred_count"] = train_manager.triple_count_by_pred
    parameters["pred_loc_count"] = train_manager.triple_count_by_pred_loc
    parameters["head_context"] = train_manager.headDict
    parameters["tail_context"] = train_manager.tailDict

    mu = ModelUtils.getModel(model_name, parameters)
    mu.set_params(parameters)
    print("Model name : ", mu.get_model_name())

    loss = LossUtils.getLoss(gamma=parameters["gamma"], model=mu, reg_type=parameters["reg_type"])

    validation = Evaluator(TripleManager(path, splits=[split_prefix + "valid", split_prefix + "train"],
                                         batch_size=parameters["batch_size"], neg_rate=parameters["nr"],
                                         corruption_mode=corruption_mode),
                           rel_anomaly_max=rel_anomaly_max, rel_anomaly_min=rel_anomaly_min,
                           batched=False)

    checkpoint_dir = folder + "Model/" + str(dataset) + "/" + model_name + "_" + split_prefix + "_" + str(point)

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
    if parameters["opt_method"] == "adagrad":
        optimizer = optim.Adagrad(
            loss.parameters(),
            **optimargs,
        )
    elif parameters["opt_method"] == "adadelta":
        optimizer = optim.Adadelta(
            loss.parameters(),
            **optimargs,
        )
    elif parameters["opt_method"] == "adam":
        optimizer = optim.Adam(
            loss.parameters(),
            **optimargs,
        )
    else:
        optimizer = optim.SGD(
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
