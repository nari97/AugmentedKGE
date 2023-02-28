from DataLoader.TripleManager import TripleManager
from Train.Evaluator import Evaluator, RankCollector
from Train.Trainer import Trainer
from Utils import ModelUtils, LossUtils, DatasetUtils, HyperparameterUtils
from ax.service.managed_loop import optimize
from ax.service.utils.best_point import get_best_raw_objective_point
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
    validation_epochs, train_times = 10, 10
    # Strategy to generate negatives.
    corruption_mode = "LCWA"
    # Metric to use.
    metric_str = "matsize"

    # Get the name of the dataset.
    dataset_name = DatasetUtils.get_dataset_name(dataset)

    print("Model:", model_name, "; Dataset:", dataset_name, "; Corruption:", corruption_mode)

    # Hyperparameters that are constant.
    # If you set weight decay, you are using L2 regularization without control.
    hyperparameters = {"batch_size": 1000, "nr": 25, "dim": 75, "dime": 75, "dimr": 75,
                       "lr": None, "momentum": None, "weight_decay": None, "opt_method": "adam", "seed": seed}

    # Loading dataset.
    start = time.perf_counter()
    path = folder + "Datasets/" + dataset_name + "/"
    # We assume we will be using Bernouilli, this is because TripleManager computes extra stuff if it is enabled. Then,
    #   below, we select whether we are using it or not.
    train_manager = TripleManager(path, splits=[split_prefix + "train"], batch_size=hyperparameters["batch_size"],
                              neg_rate=hyperparameters["nr"], corruption_mode=corruption_mode, seed=seed, use_bern=True)
    valid_manager = TripleManager(path, splits=[split_prefix + "valid", split_prefix + "train"],
                  batch_size=hyperparameters["batch_size"], neg_rate=hyperparameters["nr"],
                  corruption_mode=corruption_mode)
    end = time.perf_counter()
    print("Dataset initialization time: ", end - start)

    # These are useful when setting the models to run.
    hyperparameters["ent_total"] = train_manager.entityTotal
    hyperparameters["rel_total"] = train_manager.relationTotal
    hyperparameters["pred_count"] = train_manager.triple_count_by_pred
    hyperparameters["pred_loc_count"] = train_manager.triple_count_by_pred_loc
    hyperparameters["head_context"] = train_manager.headDict
    hyperparameters["tail_context"] = train_manager.tailDict

    # Get checkpoint file.
    checkpoint_dir = folder+"Model/"+str(dataset)+"/"+model_name+"_"+split_prefix+"_"+str(index)+"_Expl"
    checkpoint_file = os.path.join(checkpoint_dir + ".ckpt")

    params_to_optimize = [
        # Regularization.
        {"name": "lmbda", "value_type": "float", "type": "range", "bounds": [1e-4, 1.0]},
        {"name": "reg_type", "value_type": "str", "type": "choice", "values": ['L1', 'L2', 'L3'], "is_ordered":True},
        # Weight of constraints over parameters and negatives.
        {"name": "weight_constraints", "value_type": "float", "type": "range", "bounds": [1e-4, 1.0]},
        {"name": "weight_negatives", "value_type": "float", "type": "range", "bounds": [1e-4, 1.0]},
        # Margin of loss functions with margins.
        {"name": "margin", "value_type": "float", "type": "range", "bounds": [1e-4, 10.0]},
        # Norms of models that use vector norms.
        {"name": "pnorm", "value_type": "int", "type": "choice", "values": [1, 2], "is_ordered":True},
        # Whether using Bernoulli or not for sampling negatives.
        {"name": "use_bern", "value_type": "bool", "type": "choice", "values": [False, True], "is_ordered":True},
    ]

    def train_evaluate(parameterization, saving_file=None):
        print('Trying hyperparameters:', parameterization)
        # Update the existing hyperparameters with the ones provided.
        hyperparameters.update(parameterization)

        # Loading model and loss.
        start = time.perf_counter()
        mu = ModelUtils.getModel(model_name, hyperparameters)
        mu.set_hyperparameters(hyperparameters)
        print("Model name : ", mu.get_model_name())

        # Use always margin loss.
        loss = LossUtils.getLoss(model=mu, loss_str='margin', margin=hyperparameters["margin"],
                                 reg_type=hyperparameters["reg_type"], neg_weight=hyperparameters["weight_negatives"])

        validation = Evaluator(valid_manager, rel_anomaly_max=rel_anomaly_max, rel_anomaly_min=rel_anomaly_min,
                               batched=False)

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

        # Restart the managers.
        train_manager.restart()

        # Let's train!
        trainer = Trainer(loss=loss, train=train_manager, validation=validation, train_times=train_times,
                          save_steps=validation_epochs, optimizer=optimizer)
        trainer.run(metric_str=metric_str)
        end = time.perf_counter()
        print("Time elapsed during training: ", end - start)

        # Get collector.
        current_collector = RankCollector()
        current_collector.load(loss.model.ranks, loss.model.totals)

        # Save model.
        if saving_file is not None:
            torch.save(loss.model, saving_file)

        # Return MR.
        return current_collector.get_metric().get()

    # Optimize using Ax.
    best_parameters, values, experiment, model = optimize(
        parameters=params_to_optimize,
        evaluation_function=train_evaluate,
        objective_name=metric_str,
        minimize=True,
        # Note that this does not guarantee reproducibility. See:
        #   https://github.com/facebook/Ax/issues/151#issuecomment-524446967
        random_seed=seed,
    )

    # The previous best hyperparameter values may be predictions; we want the actual values found in the "raw" data.
    best_parameters, values = get_best_raw_objective_point(experiment)
    print('Best hyperparameters found:', best_parameters)
    # Train again with best hyperparameter values and save.
    train_evaluate(parameterization=best_parameters, saving_file=checkpoint_file)


if __name__ == '__main__':
    run()
