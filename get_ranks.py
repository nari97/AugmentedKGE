from DataLoader.TripleManager import TripleManager
from Train.TopKEvaluator import TopKEvaluator
from Utils import DatasetUtils
import time
import os
import sys
import torch
import glob
from datetime import datetime


def run():
    # This is the main folder where AKGE is located.
    folder = sys.argv[1]
    # This is the file that contains the configuration: algo,dataset,split.
    config_file = sys.argv[2]
    # This is the line to read in the file.
    index = int(sys.argv[3])

    shuffle_prefix = 0

    # These parameters will be constant.
    # All predicates will be considered.
    rel_anomaly_min, rel_anomaly_max = 0, 1.0
    corruption_mode = "LCWA"
    k = 1000

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
            # last_inspected = int(last_inspected)
            if line_count == index + 1:
                break

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

    print('Creating triple manager... --', datetime.now())

    splits_of_interest = [
        str(shuffle_prefix)+"_"+split_prefix + "test",
        split_prefix + "test",
        split_prefix + "valid",
        split_prefix + "train"]

    manager = TripleManager(path, splits=splits_of_interest, corruption_mode=corruption_mode)

    print('Triple manager created! Evaluating... --', datetime.now())

    evaluator = TopKEvaluator(manager, rel_anomaly_max=rel_anomaly_max, rel_anomaly_min=0)
    with torch.no_grad():
        evaluator.evaluate_top_k(model, ranks_tsv_file, k=k)

    end = time.perf_counter()

    print("Test time: ", end - start)


if __name__ == '__main__':
    run()
