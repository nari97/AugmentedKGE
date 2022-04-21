from DataLoader.TripleManager import TripleManager
from Train.Evaluator import Evaluator
import torch
import time
import sys
import glob
import os
import jsonpickle
import math

def test(model_name, dataset, corruption_mode, type = "test", use_gpu = False):
    #folder = sys.argv[1]
    #model_name = sys.argv[2]
    #dataset = int(sys.argv[3])
    #type = sys.argv[4] # valid or test
    #corruption_mode = "LCWA"

    folder = ""
    # model_name = "trans"
    # dataset = 6
    # type = "test"
    # corruption_mode = "LCWA"

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
        manager = TripleManager(path, splits=["new_valid", "new_train"], corruption_mode=corruption_mode)
        rel_anomaly_max = .75
        evaluators["MRH"] = {'metric_str':"mrh", 'rel_anomaly_max':.75, 'rel_anomaly_min':0}
    elif type == "test":
        manager = TripleManager(path, splits=["new_test", "new_valid", "new_train"], corruption_mode=corruption_mode)
        rel_anomaly_max = 1
        evaluators["MR_Global"] = {'metric_str':'mr', 'rel_anomaly_max':1, 'rel_anomaly_min':0}
        evaluators["MRG_Global"] = {'metric_str':'mrg', 'rel_anomaly_max':1, 'rel_anomaly_min':0}
        evaluators["MRH_Global"] = {'metric_str':'mrh', 'rel_anomaly_max':1, 'rel_anomaly_min':0}

        evaluators["MR_Q1"] = {'metric_str': 'mr', 'rel_anomaly_max': 1, 'rel_anomaly_min': .75}
        evaluators["MRG_Q1"] = {'metric_str': 'mrg', 'rel_anomaly_max': 1, 'rel_anomaly_min': .75}
        evaluators["MRH_Q1"] = {'metric_str': 'mrh', 'rel_anomaly_max': 1, 'rel_anomaly_min': .75}

        evaluators["MR_Q2"] = {'metric_str': 'mr', 'rel_anomaly_max': .7499, 'rel_anomaly_min': .5}
        evaluators["MRG_Q2"] = {'metric_str': 'mrg', 'rel_anomaly_max': .7499, 'rel_anomaly_min': .5}
        evaluators["MRH_Q2"] = {'metric_str': 'mrh', 'rel_anomaly_max': .7499, 'rel_anomaly_min': .5}

        evaluators["MR_Q3"] = {'metric_str': 'mr', 'rel_anomaly_max': .4999, 'rel_anomaly_min': .25}
        evaluators["MRG_Q3"] = {'metric_str': 'mrg', 'rel_anomaly_max': .4999, 'rel_anomaly_min': .25}
        evaluators["MRH_Q3"] = {'metric_str': 'mrh', 'rel_anomaly_max': .4999, 'rel_anomaly_min': .25}

        evaluators["MR_Q4"] = {'metric_str': 'mr', 'rel_anomaly_max': .2499, 'rel_anomaly_min': 0}
        evaluators["MRG_Q4"] = {'metric_str': 'mrg', 'rel_anomaly_max': .2499, 'rel_anomaly_min': 0}
        evaluators["MRH_Q4"] = {'metric_str': 'mrh', 'rel_anomaly_max': .2499, 'rel_anomaly_min': 0}

    evaluator = Evaluator(manager, rel_anomaly_max=rel_anomaly_max, rel_anomaly_min=0, use_gpu= use_gpu)

    models = []
    collectors = []
    pending = 0
    
    #glob.glob(folder + "Model/" + str(dataset) + "/" + model_name + "*.model")
    for model_file in glob.glob(folder + "Model/" +  model_name + "*.model"):
        pending = pending+1

    #glob.glob(folder + "Model/" + str(dataset) + "/" + model_name + "*.model")
    for model_file in glob.glob(folder + "Model/"  + model_name + "*.model"):
        print('Pending:',pending)
        pending = pending-1

        file = model_file.replace('.model', '.'+type)
        # Check if the file already exists
        if os.path.isfile(file):
            # Read rc
            with open(file, 'r') as f:
               rc = jsonpickle.decode(f.read())

            # These keys come as strings instead of ints.
            rc.unique_triples_materialized = {int(k): v for k, v in rc.unique_triples_materialized.items()}
            rc.total_unique_triples = {int(k): v for k, v in rc.total_unique_triples.items()}
        else:
            #util = ModelUtils(model_name, ModelUtils.get_params(model_file))
            #print (manager.hpt, manager.tph)
            model = torch.load(model_file)
            #model.model.load_checkpoint(model_file)
            #print (model.embeddings["entity"]["e"].emb.weight.data[6])
            rc = evaluator.evaluate(model, type == "test", name = model_name, dataset = dataset)
            # Store rc
            with open(file, 'w') as f:
                f.write(jsonpickle.encode(rc))

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

        there_was_a_change = True
        while there_was_a_change:
            there_was_a_change = False
            to_remove_coll, to_remove_model = None, None

            for i in range(len(current_collectors)):
                imetric = current_collectors[i].get_metric(metric_str=ev['metric_str'])
                imodel = current_models[i]

                for j in range(i+1,len(current_collectors)):
                    jmetric = current_collectors[j].get_metric(metric_str=ev['metric_str'])
                    jmodel = current_models[j]

                    if imetric.is_improved(jmetric) and current_collectors[i].is_significant(current_collectors[j].all_ranks):
                        to_remove_coll = current_collectors[i]
                        to_remove_model = imodel
                        print("This model was better:", jmodel, "; Metrics:", imetric.get(), " -- ", jmetric.get())
                    elif jmetric.is_improved(imetric) and current_collectors[j].is_significant(current_collectors[i].all_ranks):
                        to_remove_coll = current_collectors[j]
                        to_remove_model = jmodel
                        print("This model was better:", imodel, "; Metrics:", jmetric.get(), " -- ", imetric.get())

                    if to_remove_coll:
                        print('mv ' + to_remove_model.replace('.model', '.*') + ' ' +
                              to_remove_model.replace(os.path.basename(to_remove_model), '').replace('Model/', 'Model/DiscardedValidation/'))
                        break
                if to_remove_coll:
                    break

            if to_remove_coll:
                current_collectors.remove(to_remove_coll)
                current_models.remove(to_remove_model)
                there_was_a_change = True

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
            print("Unique triples materialized:", max_collector.unique_triples_materialized,
                  " out of: ", max_collector.total_unique_triples)
            print("Total unique triples materialized:", sum(max_collector.unique_triples_materialized.values()),
                  " out of: ", sum(max_collector.total_unique_triples.values()), '; Percentage: ',
                  sum(max_collector.unique_triples_materialized.values())/sum(max_collector.total_unique_triples.values()))
            print("Ties: ", max_collector.all_ties.count(True),
                  " out of: ", len(max_collector.all_ties), '; Pencentage: ',
                  max_collector.all_ties.count(True)/len(max_collector.all_ties))
            print("Ranks below expected: ", max_collector.get_ranks_below_expected().count(True),
                  " out of: ", len(max_collector.all_ranks), '; Pencentage: ',
                  max_collector.get_ranks_below_expected().count(True)/len(max_collector.all_ranks))
        print()

    end = time.perf_counter()
    print("Time elapsed to check models:", str(end - start))