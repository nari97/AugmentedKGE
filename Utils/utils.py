import torch


def check_params(params):
    """
        Checks hyperparameters for validity

        Args:
            params (dict) : Dictionary containing the hyperparameters of the experiment
            

        Returns:
    """

    flag = False
    if "nbatches" not in params:
        print("Number of batches are missing")
        flag = True
    if "nr" not in params:
        print ("Negative rate is missing")
        flag = True
    if "lr" not in params:
        print ("Learning rate is missing")
        flag = True
    if "wd" not in params:
        print ("Weight decay is missing")
        flag = True
    if "m" not in params:
        print ("Momentum is missing")
        flag = True
    if "trial_index" not in params:
        print ("Trial index is missing")
        flag = True
    if "dim" not in params or "dim_e" not in params or "dim_r" not in params:
        print ("Dimensions for embeddings are missing")
        flag = True
    if "pnorm" not in params:
        print ("Learning rate is missing")
        flag = True
    if "gamma" not in params:
        print ("Gamma is missing")
        flag = True
    if "inner_norm" not in params:
        print ("Inner norm is missing")
        flag = True

    if flag:
        exit(0)
