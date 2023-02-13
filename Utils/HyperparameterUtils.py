from scipy.stats import qmc
from torch import optim


def get_points(d=6, m=7, seed=None):
    # d is how many hyperparameters we have.
    sampler = qmc.Sobol(d=d, seed=seed)
    # This returns an array of size 2^m with arrays of size d.
    return sampler.random_base2(m=m)


# This method gets a point and decodes into hyperparameter values that are added to hyperparameters.
def decode(hyperparameters, point):
    # Regularization.
    hyperparameters["lmbda"] = scale(point[0], 1e-4, 1.0)
    hyperparameters["reg_type"] = scale_option(['L1', 'L2', 'L3'], point[1])

    # Weight of constraints over parameters.
    hyperparameters["weight_constraints"] = scale(point[2], 1e-4, 1.0)

    # Margin of loss functions with margins.
    hyperparameters["margin"] = scale(point[3], 1e-4, 10.0)
    hyperparameters["other_margin"] = scale(point[4], 1e-4, 10.0)

    # Norms of models that use vector norms.
    hyperparameters["pnorm"] = scale_option([1, 2], point[5])


# This method gets an array of options and a value between 0 and 1; one option is returned based on the value, e.g.,
#   if options=['A', 'B', 'C'] and value=.75, option 'C' is returned value is between .66 and 1.
def scale_option(options, value):
    i, step, selected = 0, 1/len(options), None
    while selected is None:
        min_interval, max_interval = i*step, (i+1)*step
        if (value >= min_interval and value < max_interval) or max_interval > 1:
            selected = i
            break
        i += 1
    return options[i]


# This method applies feature scaling from [0, 1) to [a, b).
def scale(value, a, b):
    return a + (b-a)*value


def get_optimizer(hyperparameters, loss):
    optimargs = {k: v for k, v in dict(
        lr=hyperparameters["lr"],
        weight_decay=hyperparameters["weight_decay"],
        momentum=hyperparameters["momentum"], ).items() if v is not None}

    optimizer = None
    if hyperparameters["opt_method"] is "adagrad":
        optimizer = optim.Adagrad(
            loss.parameters(),
            **optimargs,
        )
    elif hyperparameters["opt_method"] is "adadelta":
        optimizer = optim.Adadelta(
            loss.parameters(),
            **optimargs,
        )
    elif hyperparameters["opt_method"] is "adam":
        optimizer = optim.Adam(
            loss.parameters(),
            **optimargs,
        )
    else:
        optimizer = optim.SGD(
            loss.parameters(),
            **optimargs,
        )
    return optimizer
