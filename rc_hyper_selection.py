from DataLoader.TripleManager import TripleManager
from Train.Evaluator import Evaluator, RankCollector
from Train.Trainer import Trainer
from Utils import ModelUtils, LossUtils, DatasetUtils, HyperparameterUtils
import time
import os
import sys
import torch


def run():
    # This is the main folder where AKGE is located.
    folder = sys.argv[1]
    # This is the file that contains the configuration: algo,dataset,split.
    config_file = sys.argv[2]
    # This is the line to read in the file.
    index = int(sys.argv[3])
    # This seed will be used to generate the same points with Sobol sequence (the answer to the ultimate question).
    seed = 42

    # Read file.
    with open(config_file) as f:
        line_count = 0
        while True:
            line_count += 1
            line = f.readline()
            # Get configuration.
            model_name, dataset, split_prefix, last_inspected = line.split(',')
            split_prefix = split_prefix.strip()
            dataset = int(dataset)
            # This is the last point in the Sobol sequence that was inspected. Change this to resume.
            last_inspected = int(last_inspected)
            if line_count == index+1:
                break

    # These parameters will be constant.
    # All predicates will be considered.
    rel_anomaly_min, rel_anomaly_max = 0, 1.0
    # Validation and max epochs.
    validation_epochs, train_times = 25, 25
    # Strategy to generate negatives.
    corruption_mode = "LCWA"
    # Metric to use.
    metric_str = "mr"

    # Get the name of the dataset.
    dataset_name = DatasetUtils.get_dataset_name(dataset)

    print("Model:", model_name, "; Dataset:", dataset_name, "; Corruption:", corruption_mode)

    # Hyperparameters that are constant.
    # If you set weight decay, you are using L2 regularization without control.
    hyperparameters = {"batch_size": 10000, "nr": 25, "dim": 250, "dime": 250, "dimr": 250,
                       "lr": None, "momentum": None, "weight_decay": None, "opt_method": "adam", "seed": seed}

    # Loading dataset.
    start = time.perf_counter()
    path = folder + "Datasets/" + dataset_name + "/"
    train_manager = TripleManager(path, splits=[split_prefix + "train"], batch_size=hyperparameters["batch_size"],
                                  neg_rate=hyperparameters["nr"], corruption_mode=corruption_mode, seed=seed)
    end = time.perf_counter()
    print("Dataset initialization time: ", end - start)

    # These are useful when setting the models to run.
    hyperparameters["ent_total"] = train_manager.entityTotal
    hyperparameters["rel_total"] = train_manager.relationTotal
    hyperparameters["pred_count"] = train_manager.triple_count_by_pred
    hyperparameters["pred_loc_count"] = train_manager.triple_count_by_pred_loc
    hyperparameters["head_context"] = train_manager.headDict
    hyperparameters["tail_context"] = train_manager.tailDict

    # Hyperparameters to find optimal values.
    # TODO Add more dimensions and establish which position corresponds to each hyperparameter value. Note that in the
    #   future we may need to add more hyperparameters, so we need to account for them now. I think using d=15 should
    #   work. Make sure each position is established in HyperparameterUtils.
    points = HyperparameterUtils.get_points(d=7, m=7, seed=seed)

    # Get checkpoint file.
    checkpoint_dir = folder + "Model/" + str(dataset) + "/" + model_name + "_" + split_prefix + "_" + str(index)
    checkpoint_file = os.path.join(checkpoint_dir + ".ckpt")

    selected, current = None, last_inspected
    # If a checkpoint exists, we are resuming, so let's continue where we left it.
    if os.path.exists(checkpoint_file):
        # We need the hyperparameters first.
        model_hyperparameters = torch.load(checkpoint_file).pop("hyperparameters")
        model = ModelUtils.getModel(model_name, model_hyperparameters)
        model.set_hyperparameters(model_hyperparameters)
        model.load_checkpoint(checkpoint_file)
        loss = LossUtils.getLoss(margin=model_hyperparameters["margin"],
                                 other_margin=model_hyperparameters["other_margin"],
                                 model=model, reg_type=model_hyperparameters["reg_type"])
        selected = loss
        selected.model = model

    for current in range(0, len(points)):
        point = points[current]
        current += 1
        print('Current point:', current)
        HyperparameterUtils.decode(hyperparameters, point)

        # Copy to print but not all hyperparameters.
        hyper_copy = dict(hyperparameters)
        hyper_copy.pop("pred_count", None)
        hyper_copy.pop("pred_loc_count", None)
        hyper_copy.pop("head_context", None)
        hyper_copy.pop("tail_context", None)
        print("Trying hyperparameters:", hyper_copy)

        # Loading model and loss.
        start = time.perf_counter()
        mu = ModelUtils.getModel(model_name, hyperparameters)
        mu.set_hyperparameters(hyperparameters)
        print("Model name : ", mu.get_model_name())

        loss = LossUtils.getLoss(margin=hyperparameters["margin"], other_margin=hyperparameters["other_margin"],
                                 model=mu, reg_type=hyperparameters["reg_type"])

        validation = Evaluator(TripleManager(path, splits=[split_prefix + "valid", split_prefix + "train"],
                                             batch_size=hyperparameters["batch_size"], neg_rate=hyperparameters["nr"],
                                             corruption_mode=corruption_mode),
                               rel_anomaly_max=rel_anomaly_max, rel_anomaly_min=rel_anomaly_min, batched=False)

        # Initialize model from scratch
        loss.model.initialize_model()

        end = time.perf_counter()
        print("Model initialization time: ", end - start)

        start = time.perf_counter()

        # Get optimizer.
        optimizer = HyperparameterUtils.get_optimizer(hyperparameters, loss)

        # Let's inform the Trainer whether the loss function is paired or unpaired.
        if loss.is_pairwise:
            train_manager.pairing_mode = "Paired"
        else:
            train_manager.pairing_mode = "Unpaired"
        # Whether to use Bernoulli or uniform when corrupting heads/tails.
        train_manager.use_bern = hyperparameters["use_bern"]

        # Let's train!
        trainer = Trainer(loss=loss, train=train_manager, validation=validation, train_times=train_times,
                              save_steps=validation_epochs, optimizer=optimizer)
        trainer.run(metric_str=metric_str)
        end = time.perf_counter()
        print("Time elapsed during training: ", end - start)

        if selected is None:
            # Nothing selected yet, just save.
            selected = loss
            selected.model.save_checkpoint(path=checkpoint_file, epoch=validation_epochs)
        else:
            # Get current collector.
            current_collector = RankCollector()
            current_collector.load(loss.model.ranks, loss.model.totals)

            # Get other collector.
            other_collector = RankCollector()
            other_collector.load(selected.model.ranks, selected.model.totals)

            # If false, it means it was not improved. If true, it was improved and significant.
            if current_collector.stop_train(other_collector, metric_str=metric_str):
                selected = loss
                selected.model.save_checkpoint(path=checkpoint_file, epoch=validation_epochs)


if __name__ == '__main__':
    run()
