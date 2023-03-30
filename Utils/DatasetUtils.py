def get_dataset_name(dataset):
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
    if dataset == 8:
        dataset_name = "BioKG"
    if dataset == 9:
        dataset_name = "Hetionet"
    if dataset == 10:
        dataset_name = "SNOMED"
    return dataset_name
