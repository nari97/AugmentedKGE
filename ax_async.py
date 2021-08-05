import sys
import os
import os.path
from os import path
import glob
import pickle
from ax.service.ax_client import AxClient
from ax.plot.render import plot_config_to_html
from ax.utils.report.render import render_report_elements
from ax.service.utils.best_point import get_best_raw_objective_point

def get_params():
    parameters = {}
    parameters['lr'] = {"name": "lr", "type": "range", "bounds": [1e-10, 1.0], "value_type": "float"}
    parameters['pnorm'] = {"name": "pnorm", "type": "choice", "values": [1, 2], "value_type": "int"}
    parameters['nr'] = {"name": "nr", "type": "range", "bounds": [1, 50], "value_type": "int"}
    parameters['gamma'] = {"name": "gamma", "type": "range", "bounds": [.01, 10.0], "value_type": "float"}
    parameters['dim'] = {"name": "dim", "type": "range", "bounds": [50, 250], "value_type": "int"}
    parameters['dime'] = {"name": "dime", "type": "range", "bounds": [50, 250], "value_type": "int"}
    parameters['dimr'] = {"name": "dimr", "type": "range", "bounds": [50, 250], "value_type": "int"}
    parameters['bern'] = {"name": "bern", "type": "choice", "values": [False, True], "value_type": "bool"}
    parameters['norm'] = {"name": "norm", "type": "choice", "values": [False, True], "value_type": "bool"}
    parameters['nbatches'] = {"name": "nbatches", "type": "range", "bounds": [1, 250], "value_type": "int"}
    parameters['wd'] = {"name": "wd", "type": "range", "bounds": [1e-10, .1], "value_type": "float", "log_scale": True}
    parameters['m'] = {"name": "m", "type": "range", "bounds": [.5, 1.0], "value_type": "float"}
    return parameters

def run():
    #folder = sys.argv[1]
    folder = ""

    parameters = get_params()

    #models = ["transe", "transh", "transd", "rescal", "distmult", "complex", "hole", "simple", "analogy", "rotate", "transr"]
    #datasets = [0, 1, 2, 3, 4, 5, 6, 7]
    # TODO Remove!
    models = ["transe"]
    datasets = [6]

    commands = []
    for d in datasets:
        for m in models:
            ax_file = folder + "Ax/" + m + "_" + str(d) + ".ax"

            if not os.path.exists(ax_file):
                # Create client
                ax_client = AxClient(enforce_sequential_optimization=False,verbose_logging=False)
                ax_client.create_experiment(
                    name=m + "_" + str(d),
                    parameters=get_parameters(parameters, m),
                    parameter_constraints=get_parameter_constraints(m),
                    objective_name="mrh",
                    minimize=True,
                )
                ax_client.save_to_json_file(ax_file)

            # Load client
            ax_client = AxClient.load_from_json_file(ax_file, verbose_logging=False)

            # Read trials.
            trials, trial_params = [], {}
            for trial_file in glob.glob(folder + "Ax/" + m + "_" + str(d) + "_*.trial"):
                with open(trial_file, 'rb') as f:
                    trial = pickle.load(f)
                trials.append(trial['trial_index'])
                trial_index = trial.pop('trial_index', None)
                trial_params[trial_index]=trial
            trials.sort()
            
            for t in trials:
                # It seems this attaching is just registering.
                if ax_client.experiment.trials.__len__() < (t + 1):
                    print('Missing experiment: ', ax_file, '; Trial index: ', t)

                # Report results when present
                result_file = folder + "Ax/" + m + "_" + str(d) + "_" + str(t) + ".result"
                if path.exists(result_file):
                    # Get reported results
                    with open(result_file, 'rb') as f:
                        result = pickle.load(f)
                    if result['mrh'] == 1.0:
                        print('Result of trial ', result['trial_index'], ' for model ', ax_file, ' was 1.0, weird! I am not completing this...')
                        continue
                    if not ax_client.experiment.trials[result['trial_index']].status.is_completed:
                        ax_client.complete_trial(result['trial_index'], raw_data={'mrh': result['mrh']})

                # Report failures when present
                fail_file = folder + "Ax/" + m + "_" + str(d) + "_" + str(t) + ".fail"
                if path.exists(fail_file):
                    # Get reported results
                    with open(fail_file, 'rb') as f:
                        failure = pickle.load(f)
                    if not ax_client.experiment.trials[failure['trial_index']].status.is_failed:
                        ax_client.log_trial_failure(failure['trial_index'])
            ax_client.save_to_json_file(ax_file)

            ax_client = AxClient.load_from_json_file(ax_file, verbose_logging=False)

            pending_trials = []
            pending, done = 0, 0
            for t in ax_client.experiment.trials.keys():
                if ax_client.experiment.trials[t].status.is_running:
                    pending = pending + 1
                    pending_trials.append(t)
                else:
                    done = done + 1

            print(m, d, '; Pending: ', pending, ' (', pending_trials, '); Done: ', done)

            if pending == 0:
                compute_next = False

                try:
                    predictions = ax_client.get_model_predictions()
                except:
                    predictions = None

                if predictions is None or not predictions.keys():
                    compute_next = True
                    print('No model predictions')
                else:
                    # Get current, real mean.
                    current_best_parameters, values = get_best_raw_objective_point(ax_client.experiment)
                    current_mean, current_sem = values['mrh']

                    # Get new, expected mean.
                    new_best_parameters, values = ax_client.get_best_parameters()
                    means, covariances = values
                    new_mean = means['mrh']

                    if current_mean <= new_mean:
                        print('Experiment is over! Best model: ', current_best_parameters, '; MRH:', current_mean)
                        plot = ax_client.get_optimization_trace(objective_optimum=1.0)
                        with open(folder + "Ax/" + m + "_" + str(d) + '.html', 'w') as outfile:
                           outfile.write(render_report_elements(
                                m + "_" + str(d),
                                html_elements=[plot_config_to_html(plot)],
                                header=False,
                            ))
                    else:
                        print('Current MRH:',current_mean,'; Expected MRH:',new_mean)
                        # Request next trial, one by one now.
                        command = next_trials(ax_client, folder, m, d, n=1)
                        commands.append(command)

                if compute_next:
                    command = next_trials(ax_client, folder, m, d)
                    commands.append(command)

            # Save client
            ax_client.save_to_json_file(ax_file)

    # Print all commands at once.
    for c in commands:
        print(c)

def next_trials(ax_client, folder, m, d, n=None):
    if n is None:
        num_trials = trials_in_parallel(ax_client)
    else:
        num_trials = n

    # Create initial parameter assignments
    init, end = -1, -1
    for i in range(num_trials):
        trial_params, trial_index = ax_client.get_next_trial()
        if init == -1:
            init = trial_index
        end = trial_index
        # Add index and save
        trial_params["trial_index"] = trial_index
        with open(folder + "Ax/" + m + "_" + str(d) + "_" + str(trial_index) + ".trial", 'wb') as f:
            pickle.dump(trial_params, f, pickle.HIGHEST_PROTOCOL)

    return 'sbatch --array=' + (str(init) + '-' + str(end) if end > init else str(init)) + \
           (' --mem=30000 ' if m == "rescal" else '') + ' run_train.sh /home/crrvcs/KGE/ ' + m + ' ' + str(d)

def trials_in_parallel(ax_client):
    (num_trials, max_setting) = ax_client.get_max_parallelism()[0]
    if max_setting == -1:
        return num_trials
    else:
        return max_setting

def get_parameters(parameters, model_name):
    params = ["lr", "nr", "bern", "nbatches", "wd", "m"]
    if model_name == "transd":
        params = params + ["dime", "dimr"]
    elif model_name == "transr":
        params = params + ["dime" , "dimr"]
    else:
        params = params + ["dim"]
    if model_name.startswith("trans") or model_name == "hole" or model_name == "rotate":
        params = params + ["pnorm", "gamma", "norm"]
    if model_name == "rescal" or model_name == 'distmult' or model_name == 'complex' or model_name == 'simple' or model_name == 'analogy':
        params = params + ["gamma"]
    ret = []
    for p in params:
        ret.append(parameters[p]);
    return ret

def get_parameter_constraints(model_name):
    ret = []
    if model_name == "transd":
        # TransD does not work when dimr > dime
        ret.append("dime >= dimr")
    if model_name == "transr":
        ret.append("dime != dimr")
    return ret

if __name__ == '__main__':
    run()