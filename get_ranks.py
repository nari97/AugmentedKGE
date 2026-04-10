from Models import BoxE
from DataLoader.TripleManager import TripleManager
from Train.TopKEvaluator import TopKEvaluator
from Utils import DatasetUtils
import time
import os
import sys
import torch
import glob


def run():
    # This is the main folder where AKGE is located.
    folder = sys.argv[1]
    # This is the configuration number. See below.
    index = int(sys.argv[2])

    # These parameters will be constant.
    # All predicates will be considered.
    rel_anomaly_min, rel_anomaly_max = 0, 1.0
    corruption_mode = "LCWA"
    k = 1000

    def get_scenario():
        i, split_prefix = 0, ''
        for dataset in range(0, 11):
            if dataset == 3:
                split_prefix = 'resplit_'
            else:
                split_prefix = ''

            for model_name in ['boxe', 'complex', 'hake_both', 'hole', 'quate',
                               'rotate', 'rotpro', 'toruse_eL2', 'transe', 'tucker']:
                if i == index:
                    return dataset, model_name, split_prefix, 0
                else:
                    i = i + 1

    dataset, model_name, split_prefix, shuffle_prefix = get_scenario()

    # Get the name of the dataset.
    dataset_name = DatasetUtils.get_dataset_name(dataset)

    print("Model:", model_name, "; Dataset:", dataset_name,
          "; Corruption:", corruption_mode, '; Shuffle:', shuffle_prefix)

    # Loading model and dataset.
    start = time.perf_counter()

    # Get checkpoint file. It must exist!

    for model_file in glob.glob(
            folder + "Model/" + str(dataset) + "/" +
            model_name + "_" + split_prefix + "_*.model"):
        pass

    path = folder + "Datasets/" + dataset_name + "/"
    ranks_tsv_file = folder+"Model/"+str(dataset)+"/"+model_name+\
                     "_"+split_prefix+"_"+str(shuffle_prefix)+"_ranks.tsv"

    # If this exists, we are done; otherwise, let's go for it.
    if not os.path.exists(model_file):
        raise Exception("Error: model ", model_file, " does not exists!")
    else:
        print("Model exists! Loading...")
        model = torch.load(model_file, weights_only=False)
        print("Loaded!")

    splits_of_interest = [
        str(shuffle_prefix)+"_"+split_prefix + "test",
        split_prefix + "test",
        split_prefix + "valid",
        split_prefix + "train"]
    manager = TripleManager(path, splits=splits_of_interest, corruption_mode=corruption_mode)

    evaluator = TopKEvaluator(manager, rel_anomaly_max=rel_anomaly_max, rel_anomaly_min=0)
    with torch.no_grad():

        evaluator.evaluate_top_k(model, ranks_tsv_file, k=k)
    end = time.perf_counter()
    print("Test time: ", end - start)


if __name__ == '__main__':
    run()
