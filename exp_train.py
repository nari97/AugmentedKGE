from DataLoader.TripleManager import TripleManager
from Train.Evaluator import Evaluator, RankCollector
from Train.Trainer import Trainer
from Utils import LossUtils, DatasetUtils, HyperparameterUtils
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
    validation_epochs, train_times = 50, 10000
    corruption_mode = "LCWA"
    # Metric to use.
    metric_str = "matsize"

    # Get the name of the dataset.
    dataset_name = DatasetUtils.get_dataset_name(dataset)

    print("Model:", model_name, "; Dataset:", dataset_name, "; Corruption:", corruption_mode)

    # Loading model and dataset.
    start = time.perf_counter()

    # Get checkpoint file. It must exist!
    checkpoint_dir = folder+"Model/"+str(dataset)+"/"+model_name+"_"+split_prefix+"_"+str(index)+"_Expl"
    checkpoint_file = os.path.join(checkpoint_dir + ".ckpt")
    valid_file = os.path.join(checkpoint_dir + ".valid")
    model_file = os.path.join(checkpoint_dir + ".model")
    path = folder + "Datasets/" + dataset_name + "/"

    # If this exists, we are done; otherwise, let's go for it.
    if not os.path.exists(model_file):
        model = torch.load(checkpoint_file)
        init_epoch = model.epoch
        hyperparameters = model.get_hyperparameters()

        # We assume we will be using Bernouilli, this is because TripleManager computes extra stuff if it is enabled. Then,
        #   below, we select whether we are using it or not.
        train_manager = TripleManager(path, splits=[split_prefix + "train"], batch_size=hyperparameters["batch_size"],
                                      neg_rate=hyperparameters["nr"], corruption_mode=corruption_mode, seed=seed,
                                      use_bern=True)
        end = time.perf_counter()
        print("Model and dataset initialization time: ", end - start)

        start = time.perf_counter()
        # Use always margin loss.
        loss = LossUtils.getLoss(model=model, loss_str='margin', margin=hyperparameters["margin"],
                                 reg_type=hyperparameters["reg_type"], neg_weight=hyperparameters["weight_negatives"])

        valid_manager = TripleManager(path, splits=[split_prefix + "valid", split_prefix + "train"],
                                      batch_size=hyperparameters["batch_size"], neg_rate=hyperparameters["nr"],
                                      corruption_mode=corruption_mode)

        validation = Evaluator(valid_manager, rel_anomaly_max=rel_anomaly_max,
                               rel_anomaly_min=rel_anomaly_min, batched=False)
        end = time.perf_counter()
        print("Validation initialization time: ", end - start)

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

        # Restart the manager.
        train_manager.restart()

        # load valid function.
        def load_valid():
            valid = None
            # There is a validation model, let's load it!
            if os.path.exists(valid_file):
                valid = torch.load(valid_file)
            return valid

        # Save valid function.
        def save_valid():
            torch.save(loss.model, valid_file)

        # Save the current checkpoint.
        def save_checkpoint():
            torch.save(loss.model, checkpoint_file)

        # Let's train!
        trainer = Trainer(loss=loss, train=train_manager, validation=validation, train_times=train_times,
                          save_steps=validation_epochs, optimizer=optimizer, load_valid=load_valid,
                          save_valid=save_valid, save_checkpoint=save_checkpoint, init_epoch=init_epoch)
        trainer.run(metric_str=metric_str)
        end = time.perf_counter()
        print("Time elapsed during training: ", end - start)

        # We are done! Rename checkpoint to model.
        if not os.path.exists(model_file):
            os.rename(valid_file, model_file)
            os.remove(checkpoint_file)
    else:
        print("Model exists! Loading...")
        model = torch.load(model_file)
        print("Loaded!")

    start = time.perf_counter()

    # Run test and report metrics!
    evaluators = {}

    rel_anomaly_max = 1
    for (suffix, ramax, ramin) in [('Global', 1, 0), ('Q1', 1, .75), ('Q2', .7499, .5), ('Q3', .499, .25),
                                   ('Q4', .249, 0)]:
        evaluators["MRR_"+suffix] = {'metric_str': 'mrr', 'rel_anomaly_max': ramax, 'rel_anomaly_min': ramin}
        evaluators["MR_"+suffix] = {'metric_str': 'mr', 'rel_anomaly_max': ramax, 'rel_anomaly_min': ramin}
        evaluators["GMR_"+suffix] = {'metric_str': 'gmr', 'rel_anomaly_max': ramax, 'rel_anomaly_min': ramin}
        evaluators["WMR_"+suffix] = {'metric_str': 'wmr', 'rel_anomaly_max': ramax, 'rel_anomaly_min': ramin}
        evaluators["WGMR_"+suffix] = {'metric_str': 'wgmr', 'rel_anomaly_max': ramax, 'rel_anomaly_min': ramin}
        evaluators["MatSize_"+suffix] = {'metric_str': 'matsize', 'rel_anomaly_max': ramax, 'rel_anomaly_min': ramin}

    evaluator = Evaluator(TripleManager(path,
                                splits=[split_prefix + "test", split_prefix + "valid", split_prefix + "train"],
                                corruption_mode=corruption_mode), rel_anomaly_max=rel_anomaly_max, rel_anomaly_min=0)
    with torch.no_grad():
        main_collector = evaluator.evaluate(model, materialize=False)
    end = time.perf_counter()
    print("Test time: ", end - start)

    print('Length of main ranks:', len(main_collector.all_ranks))
    print('Length of main ties:', len(main_collector.all_ties))

    for k in evaluators.keys():
        print("Evaluator:", k)

        ev = evaluators[k]

        current_collector = main_collector.prune(ev['rel_anomaly_max'], ev['rel_anomaly_min'])
        if len(current_collector.all_ranks) == 0:
            print('No ranks!')
            print()
            continue

        print('Length of ranks:', len(current_collector.all_ranks))
        print('Length of ties:', len(current_collector.all_ties))

        metric = current_collector.get_metric(metric_str=ev['metric_str'])
        adjusted = 1.0 - (metric.get() / current_collector.get_expected(metric_str=ev['metric_str']).get())

        print("Metric:", metric.get())
        print("Expected:", current_collector.get_expected(metric_str=ev['metric_str']).get())
        print("Adjusted metric: ", adjusted)
        print("Ties: ", current_collector.all_ties.count(True),
              " out of: ", len(current_collector.all_ties), '; Pencentage: ',
              current_collector.all_ties.count(True) / len(current_collector.all_ties))
        print("Ranks below expected: ", current_collector.get_ranks_below_expected().count(True),
              " out of: ", len(current_collector.all_ranks), '; Pencentage: ',
              current_collector.get_ranks_below_expected().count(True) / len(current_collector.all_ranks))
        print()

if __name__ == '__main__':
    run()
