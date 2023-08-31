from DataLoader.TripleManager import TripleManager
from Train.Evaluator import Evaluator, RankCollector
from Train.Trainer import Trainer
from Utils import ModelUtils, LossUtils, DatasetUtils, HyperparameterUtils
from ax.service.ax_client import AxClient
import time
import os
import torch
import sys
import itertools


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
        config_filename = os.path.basename(f.name)
        line_count = 0
        while True:
            line_count += 1
            line = f.readline()
            # Get configuration.
            model_name, dataset, split_prefix, hyperparams = line.split(',')
            split_prefix = split_prefix.strip()
            dataset = int(dataset)
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
    metric_str = "mr"
    # Total trials indicate how many points we will inspect in the hyperparameter value optimization.
    total_trials = 20

    # Get the name of the dataset.
    dataset_name = DatasetUtils.get_dataset_name(dataset)

    print("Model:", model_name, "; Dataset:", dataset_name, "; Corruption:", corruption_mode)

    # Hyperparameters that are constant.
    # If you set weight decay, you are using L2 regularization without control.
    hyperparameters = {"batch_size": 1500, "nr": 25, "dim": 150, "dime": 150, "dimr": 150,
                       "lr": None, "momentum": None, "weight_decay": None, "opt_method": "adam", "seed": seed}

    # Loading dataset.
    start = time.perf_counter()
    path = folder + "Datasets/" + dataset_name + "/"
    # We assume we will be using Bernouilli, this is because TripleManager computes extra stuff if it is enabled. Then,
    #   below, we select whether we are using it or not.
    train_manager = TripleManager(path, splits=[split_prefix + "train"], batch_size=hyperparameters["batch_size"],
                                  neg_rate=hyperparameters["nr"], corruption_mode=corruption_mode, seed=seed,
                                  use_bern=True)
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
    checkpoint_dir = folder+"Model/"+str(dataset)+"/"+model_name+"_"+split_prefix+"_"+str(index)+"_"+config_filename
    checkpoint_file = os.path.join(checkpoint_dir + ".ckpt")
    ax_file = os.path.join(checkpoint_dir + ".ax")

    def train_evaluate(parameterization, saving_file=None):
        print('Trying hyperparameters:', parameterization)
        # Update the existing hyperparameters with the ones provided.
        hyperparameters.update(parameterization)

        # Loading model and loss.
        start = time.perf_counter()
        mu = ModelUtils.getModel(model_name, hyperparameters)
        mu.set_hyperparameters(hyperparameters)
        print("Model name : ", mu.get_model_name())

        loss = LossUtils.getLoss(model=mu, margin=hyperparameters["margin"],
                                 other_margin=hyperparameters["other_margin"], reg_type=hyperparameters["reg_type"],
                                 neg_weight=hyperparameters["weight_negatives"])

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

        # Return MR (and standard error, .0 in our case).
        return current_collector.get_metric().get(), .0

    params_to_optimize = [
        # Regularization.
        {"name": "lmbda", "value_type": "float", "type": "range", "bounds": [1e-4, 1.0]},
        {"name": "reg_type", "value_type": "str", "type": "choice", "values": ['L1', 'L2', 'L3'], "is_ordered": True},
        # Weight of constraints over parameters and negatives.
        {"name": "weight_constraints", "value_type": "float", "type": "range", "bounds": [1e-4, 1.0]},
        {"name": "weight_negatives", "value_type": "float", "type": "range", "bounds": [1e-4, 1.0]},
        # Margin of loss functions with margins.
        {"name": "margin", "value_type": "float", "type": "range", "bounds": [1e-4, 10.0]},
        {"name": "other_margin", "value_type": "float", "type": "range", "bounds": [1e-4, 10.0]},
        # Norms of models that use vector norms.
        {"name": "pnorm", "value_type": "int", "type": "choice", "values": [1, 2], "is_ordered": True},
        # Whether using Bernoulli or not for sampling negatives.
        {"name": "use_bern", "value_type": "bool", "type": "choice", "values": [False, True], "is_ordered": True},
    ]

    # If there are no hyperparameters specified, go for regular Ax optimization.
    if len(hyperparams) == 0:
        # Optimize using Ax.
        ax_client = AxClient(
            # Note that this does not guarantee reproducibility. See:
            #   https://github.com/facebook/Ax/issues/151#issuecomment-524446967
            random_seed=seed,
        )

        ax_client.create_experiment(
            name="hyperparameter_optimization_experiment",
            parameters=params_to_optimize,
            objective_name=metric_str,
            minimize=True, )

        if os.path.exists(ax_file):
            # Load for resume!
            ax_client = ax_client.load_from_json_file(filepath=ax_file)
            total_trials -= len(ax_client.experiment.trials)

        for i in range(total_trials):
            parameters, trial_index = ax_client.get_next_trial()
            ax_client.complete_trial(trial_index=trial_index, raw_data=train_evaluate(parameters))
            ax_client.save_to_json_file(filepath=ax_file)

        best_parameters, values = ax_client.get_best_parameters()
    else:
        best_parameters, best_value = None, None
        # Hyperparameters are specified, go for an exhaustive search.
        all_names, all_values = [], []
        for hyperparam in hyperparams.split(';'):
            # Find and get values.
            for params in params_to_optimize:
                if params['name'] == hyperparam:
                    all_names.append(params['name'])
                    all_values.append(params_to_optimize['values'])

        # Exhaustive search.
        for combination in itertools.product(all_values):
            parameters = {}
            for idx_p, p in enumerate(combination):
                parameters[all_names[idx_p]] = p

            metric, std = train_evaluate(parameters)

            if best_value is None or metric < best_value:
                best_parameters = parameters
                best_value = metric

    print('Best hyperparameters found:', best_parameters)

    # Train again with best hyperparameter values and save.
    train_evaluate(parameterization=best_parameters, saving_file=checkpoint_file)


if __name__ == '__main__':
    run()
