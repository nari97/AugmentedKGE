from DataLoader.TripleManager import TripleManager
from Train.Calibration import Calibrator, PlattCalibrator, IsotonicCalibrator
from Utils import DatasetUtils
import time
import sys
import torch
import glob


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
            model_name, dataset, split_prefix, last_inspected, calib_type, pos_tgt, neg_tgt_1, neg_tgt_2, neg_tgt_3, \
                neg_tgt_4 = line.replace('\\n','').split(',')
            split_prefix = split_prefix.strip()
            dataset = int(dataset)
            # This is the last point in the Sobol sequence that was inspected. Change this to resume.
            last_inspected = int(last_inspected)
            if line_count == index+1:
                break

    # Get the name of the dataset.
    dataset_name = DatasetUtils.get_dataset_name(dataset)

    print("Model:", model_name, "; Dataset:", dataset_name)

    # Loading model and dataset.
    # Get checkpoint file. It must exist!
    checkpoint_dir = folder+"Model/"+str(dataset)+"/"+model_name+"_"+split_prefix+"*_Expl"
    model_file = glob.glob(checkpoint_dir + ".model")[0]
    path = folder + "Datasets/" + dataset_name + "/"

    model = torch.load(model_file)
    print("Loaded!")

    splits = [split_prefix + "valid", split_prefix + "train"]
    valid_manager = TripleManager(path, splits=splits, corruption_mode='LCWA')
    # These are the managers sorted by priority.
    managers = [TripleManager(path, splits=splits, corruption_mode='TCLCWA'),
                TripleManager(path, splits=splits, corruption_mode='Local'),
                TripleManager(path, splits=splits, corruption_mode='Global'),
                valid_manager]

    print('Positive target:', pos_tgt)

    neg_tgts = [Calibrator.get_tensor(float(neg_tgt_1), False), Calibrator.get_tensor(float(neg_tgt_2), False),
               Calibrator.get_tensor(float(neg_tgt_3), False), Calibrator.get_tensor(float(neg_tgt_4), False)]

    print('Negative targets:', ",".join([str(t.item()) for t in neg_tgts]))

    start = time.perf_counter()

    def positive_target_correction(total_positives):
        # Proposed by Platt.
        return 1 / (total_positives + 2)

    def negative_target_correction(total_negatives):
        # Proposed by Platt.
        return 1 / (total_negatives + 2)

    if calib_type == 'platt':
        # Trying different A and B inits and pick the best! This is done by train.
        calib = PlattCalibrator(model=model, managers=managers, pos_target=float(pos_tgt), neg_targets=neg_tgts,
                                init_a_values=[-1, 0, 1], init_b_values=[-1, 0, 1])

        calib.train(positive_target_correction=positive_target_correction,
                    negative_target_correction=negative_target_correction, use_weight=True)
        print('A, B:', calib.a.item(), calib.b.item())
    elif calib_type == 'isotonic':
        calib = IsotonicCalibrator(model=model, managers=managers, pos_target=float(pos_tgt), neg_targets=neg_tgts)
        calib.train(positive_target_correction=positive_target_correction,
                    negative_target_correction=negative_target_correction, use_weight=True)

    end = time.perf_counter()
    print("Train time: ", end - start)

    print('Test...')
    start = time.perf_counter()

    splits = [split_prefix + "test", split_prefix + "valid", split_prefix + "train"]
    mode_results = {}

    # Get also the results for the validation split.
    print('Validating using corruption mode: LCWA')
    csv_header, csv_results = calib.test(valid_manager)
    mode_results["Valid (LCWA)"] = csv_results

    for corrupt_mode in ['TCLCWA', 'Local', 'Global', 'LCWA']:
        print('Testing using corruption mode:', corrupt_mode)
        test_manager = TripleManager(path, splits=splits, corruption_mode=corrupt_mode)
        csv_header, csv_results = calib.test(test_manager)
        mode_results[corrupt_mode] = csv_results

    end = time.perf_counter()
    print("Time: ", end - start)

    print('<CSVResults>')
    print('model,dataset,calib,pos,neg1,neg2,neg3,neg4,corruption,' + csv_header)
    for mode in mode_results.keys():
        print(str(model_name)+','+str(dataset)+','+str(calib_type)+','+pos_tgt+','+neg_tgt_1+','+\
              neg_tgt_2+','+neg_tgt_3+','+neg_tgt_4+','+mode+','+mode_results[mode])
    print('<\\CSVResults>')


if __name__ == '__main__':
    run()
