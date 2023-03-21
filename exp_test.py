from DataLoader.TripleManager import TripleManager
from Train.Evaluator import Evaluator
from Utils import DatasetUtils
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
    corruption_mode = "LCWA"

    # Get the name of the dataset.
    dataset_name = DatasetUtils.get_dataset_name(dataset)

    print("Model:", model_name, "; Dataset:", dataset_name, "; Corruption:", corruption_mode)

    # Loading model and dataset.
    start = time.perf_counter()

    # Get checkpoint file. It must exist!
    checkpoint_dir = folder+"Model/"+str(dataset)+"/"+model_name+"_"+split_prefix+"_"+str(index)+"_Expl"
    model_file = os.path.join(checkpoint_dir + ".model")
    path = folder + "Datasets/" + dataset_name + "/"

    # If this exists, we are done; otherwise, let's go for it.
    if not os.path.exists(model_file):
        raise Exception("Error: model ", model_file, " does not exists!")
    else:
        print("Model exists! Loading...")
        model = torch.load(model_file)
        print("Loaded!")

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
        main_collector = evaluator.evaluate(model, materialize=True, materialize_basefile=checkpoint_dir)
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
