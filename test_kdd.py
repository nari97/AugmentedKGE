from DataLoader.TripleManager import TripleManager
from Train.Evaluator import Evaluator
import torch
import time
import sys
import glob
import math


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
    type = sys.argv[3]  # valid or test

    model_name, dataset, split_prefix = get_params(index)

    corruption_mode = "LCWA"

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

    path = folder + "Datasets/" + dataset_name + "/"

    start = time.perf_counter()

    evaluators = {}
    if type == "valid":
        manager = TripleManager(path, splits=[split_prefix+"valid", split_prefix+"train"], corruption_mode=corruption_mode)
        rel_anomaly_max = .75
        evaluators["MRH"] = {'metric_str':"mrh", 'rel_anomaly_max':.75, 'rel_anomaly_min':0}
    elif type == "test":
        manager = TripleManager(path, splits=[split_prefix+"test", split_prefix+"valid", split_prefix+"train"], corruption_mode=corruption_mode)
        rel_anomaly_max = 1
        evaluators["MR_Global"] = {'metric_str':'mr', 'rel_anomaly_max':1, 'rel_anomaly_min':0}
        evaluators["GMR_Global"] = {'metric_str':'gmr', 'rel_anomaly_max':1, 'rel_anomaly_min':0}
        evaluators["WMR_Global"] = {'metric_str': 'wmr', 'rel_anomaly_max': 1, 'rel_anomaly_min': 0}
        evaluators["WGMR_Global"] = {'metric_str':'wgmr', 'rel_anomaly_max':1, 'rel_anomaly_min':0}

        evaluators["MR_Q1"] = {'metric_str': 'mr', 'rel_anomaly_max': 1, 'rel_anomaly_min': .75}
        evaluators["GMR_Q1"] = {'metric_str': 'gmr', 'rel_anomaly_max': 1, 'rel_anomaly_min': .75}
        evaluators["WMR_Q1"] = {'metric_str': 'wmr', 'rel_anomaly_max': 1, 'rel_anomaly_min': .75}
        evaluators["WGMR_Q1"] = {'metric_str': 'wgmr', 'rel_anomaly_max': 1, 'rel_anomaly_min': .75}

        evaluators["MR_Q2"] = {'metric_str': 'mr', 'rel_anomaly_max': .7499, 'rel_anomaly_min': .5}
        evaluators["GMR_Q2"] = {'metric_str': 'gmr', 'rel_anomaly_max': .7499, 'rel_anomaly_min': .5}
        evaluators["WMR_Q2"] = {'metric_str': 'wmr', 'rel_anomaly_max': .7499, 'rel_anomaly_min': .5}
        evaluators["WGMR_Q2"] = {'metric_str': 'wgmr', 'rel_anomaly_max': .7499, 'rel_anomaly_min': .5}

        evaluators["MR_Q3"] = {'metric_str': 'mr', 'rel_anomaly_max': .4999, 'rel_anomaly_min': .25}
        evaluators["GMR_Q3"] = {'metric_str': 'gmr', 'rel_anomaly_max': .4999, 'rel_anomaly_min': .25}
        evaluators["WMR_Q3"] = {'metric_str': 'wmr', 'rel_anomaly_max': .4999, 'rel_anomaly_min': .25}
        evaluators["WGMR_Q3"] = {'metric_str': 'wgmr', 'rel_anomaly_max': .4999, 'rel_anomaly_min': .25}

        evaluators["MR_Q4"] = {'metric_str': 'mr', 'rel_anomaly_max': .2499, 'rel_anomaly_min': 0}
        evaluators["GMR_Q4"] = {'metric_str': 'gmr', 'rel_anomaly_max': .2499, 'rel_anomaly_min': 0}
        evaluators["WMR_Q4"] = {'metric_str': 'wmr', 'rel_anomaly_max': .2499, 'rel_anomaly_min': 0}
        evaluators["WGMR_Q4"] = {'metric_str': 'wgmr', 'rel_anomaly_max': .2499, 'rel_anomaly_min': 0}

    evaluator = Evaluator(manager,rel_anomaly_max=rel_anomaly_max, rel_anomaly_min=0)

    models = []
    collectors = []

    with torch.no_grad():
        # This is only one file.
        for model_file in glob.glob(folder + "Model/" + str(dataset) + "/" + model_name+ "_" + split_prefix + "_" + str(index) + ".model"):
            model = torch.load(model_file)

            rc = evaluator.evaluate(model, materialize=False)
            # Remove from GPU once we are done!
            model.remove_from_gpu()

            models.append(model_file)
            collectors.append(rc)

        end = time.perf_counter()
        print("Time elapsed to load collectors:", str(end - start))

        start = time.perf_counter()

        for k in evaluators.keys():
            ev = evaluators[k]
            current_models = []
            current_collectors = []

            for i in range(len(models)):
                rc = collectors[i].prune(ev['rel_anomaly_max'], ev['rel_anomaly_min'])
                if len(rc.all_ranks) == 0:
                    continue
                current_models.append(models[i])
                current_collectors.append(rc)

            print("Evaluator:", k, "; Remaining models:", len(current_collectors))

            max, max_model, max_collector = -math.inf, None, None
            for i in range(len(current_collectors)):
                imetric = current_collectors[i].get_metric(metric_str=ev['metric_str'])
                imodel = current_models[i]

                print("Model:", imodel)
                current = 1.0 - (imetric.get()/current_collectors[i].get_expected(metric_str=ev['metric_str']).get())
                if current > max:
                    max = current
                    max_model = imodel
                    max_collector = current_collectors[i]

            if max_model is None:
                print('No models!')
                continue

            print("Best model:", max_model)
            print("Metric:", max_collector.get_metric(metric_str=ev['metric_str']).get())
            print("Expected:", max_collector.get_expected(metric_str=ev['metric_str']).get())
            print("Adjusted metric: ", max)
            if type == "test":
                #print("Unique triples materialized:", max_collector.unique_triples_materialized,
                #      " out of: ", max_collector.total_unique_triples)
                #print("Total unique triples materialized:", sum(max_collector.unique_triples_materialized.values()),
                #      " out of: ", sum(max_collector.total_unique_triples.values()), '; Percentage: ',
                #      sum(max_collector.unique_triples_materialized.values())/sum(max_collector.total_unique_triples.values()))
                print("Ties: ", max_collector.all_ties.count(True),
                      " out of: ", len(max_collector.all_ties), '; Pencentage: ',
                      max_collector.all_ties.count(True)/len(max_collector.all_ties))
                print("Ranks below expected: ", max_collector.get_ranks_below_expected().count(True),
                      " out of: ", len(max_collector.all_ranks), '; Pencentage: ',
                      max_collector.get_ranks_below_expected().count(True)/len(max_collector.all_ranks))
            print()

    end = time.perf_counter()
    print("Time elapsed to check models:", str(end - start))

if __name__ == '__main__':
    run()
