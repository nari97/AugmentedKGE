import glob
import matplotlib.pyplot as plt

def run():
    # Folder with output files.
    folder, fig_folder = 'C:/Users/crr/Desktop/Calib_Results/', 'platt/'

    # TODO Count how many models per model and dataset to see if we have missing!
    csv_header, csv_data = None, []
    for file in glob.glob(folder + '*.out'):
        # Get the job id. This is handy to identify things later on.
        split_name = file.split('_')
        job_id = split_name[len(split_name)-1]

        with open(file, 'r') as f:
            all_lines = f.readlines()

            in_csv, i = False, 0
            for index, line in enumerate(all_lines):
                line = line.replace('\n','')
                # Indicates whether we have started with the CSV part.
                if line == '<CSVResults>':
                    in_csv = True
                elif line == '<\\CSVResults>':
                    in_csv = False
                elif in_csv:
                    if i == 0:
                        csv_header = line.split(',')
                    elif i % 2 == 0:
                        # Due to a bug, each line is split into two (way to go!). Fixing here that stupid issue.
                        fixed_line = all_lines[index-1].replace('\n','')+line
                        # Some results contain commas, e.g., KS(X, Y). Take care of that.
                        fixed_line = fixed_line.replace(', ', '; ')
                        # Transform into dictionary form.
                        line_as_dict = {}
                        for idx, item in enumerate(fixed_line.split(',')):
                            line_as_dict[csv_header[idx]] = item

                        # Dataset 4 is problematic; skip!
                        if line_as_dict['dataset'] != '4' and line_as_dict['calib'] == 'platt' and line_as_dict['neg1'] == '0':
                                # Also, let's check Platt results only using binary targets.
                                # and line_as_dict['calib'] == 'platt' and line_as_dict['neg1'] == '0'
                            # Add derived fields: TPR=tp/(tp+fn); TNR=tn/(tn+fp); BA=(TPR+TNR)/2; BM=TPR+TNR-1.
                            tp, fp, fn, tn = float(line_as_dict['tp']), float(line_as_dict['fp']), \
                                float(line_as_dict['fn']), float(line_as_dict['tn'])
                            line_as_dict['TPR'] = str(tp/(tp+fn))
                            line_as_dict['TNR'] = str(tn/(tn+fp))
                            line_as_dict['BA'] = str((float(line_as_dict['TPR']) + float(line_as_dict['TNR'])) / 2)
                            line_as_dict['BM'] = str(float(line_as_dict['TPR']) + float(line_as_dict['TNR']) - 1)
                            line_as_dict['job_id'] = job_id

                            # Store in the data.
                            csv_data.append(line_as_dict)
                    i += 1

    # For the same model and dataset, find best calibration model using LCWA based on R2.
    best_calib_models, best_check, corruption_to_check = {}, 'r2', 'Valid (LCWA)' # LCWA, Valid (LCWA)
    for data in csv_data:
        if data['corruption'] == corruption_to_check:
            model_dataset = data['model']+'_'+data['dataset']

            def filter(x):
                if best_check == 'pearson':
                    # Change sign!
                    return -1*float(x.replace('(', '').split('; ')[0])
                else:
                    return float(x)

            if model_dataset not in best_calib_models.keys() or \
                    filter(data[best_check]) > filter(best_calib_models[model_dataset][best_check]):
                best_calib_models[model_dataset] = data

    # For the best model identified, we will store their results by strategy. We could do it by line, but we will do it
    #   using the job id.
    calib_models_other_strategies = {'Valid (LCWA)': {}, 'LCWA': {}, 'Global': {}, 'Local': {}, 'TCLCWA': {}}
    for calib_model_data in best_calib_models.values():
        for data in csv_data:
            if data['job_id'] == calib_model_data['job_id']:
                corrupt = data['corruption']
                model_dataset = data['model'] + '_' + data['dataset']
                calib_models_other_strategies[corrupt][model_dataset] = data

    def model_pretty_name(model_name, long_output=True):
        if model_name == 'boxe':
            return 'BoxE' if long_output else 'Bo'
        if model_name == 'complex':
            return 'ComplEx' if long_output else 'Co'
        if model_name == 'hake_both':
            return 'HAKE' if long_output else 'HA'
        if model_name == 'hole':
            return 'HolE' if long_output else 'Ho'
        if model_name == 'quate':
            return 'QuatE' if long_output else 'Qu'
        if model_name == 'rotate':
            return 'RotatE' if long_output else 'Rt'
        if model_name == 'rotpro':
            return 'RotPro' if long_output else 'RP'
        if model_name == 'toruse_eL2':
            return 'TorusE' if long_output else 'To'
        if model_name == 'transe':
            return 'TransE' if long_output else 'Tr'
        if model_name == 'tucker':
            return 'TuckER' if long_output else 'Tu'

    def plot_results(corrupt, key, calib_models):
        data_by_model = {}
        for calib_model_data in calib_models:
            model = calib_model_data['model']
            if model not in data_by_model.keys():
                data_by_model[model] = []
            value = calib_model_data[key]

            if key == 'pearson':
                value = value.replace('(', '').split('; ')[0]

            data_by_model[model].append(float(value))

        data_ticks, data_tick_labels, data_to_plot = [], sorted(list(data_by_model.keys())), []
        for i in range(len(data_tick_labels)):
            data_ticks.append(i + 1)
            data_to_plot.append(data_by_model[data_tick_labels[i]])

        y_lim_bottom, y_lim_top = 0, 1

        if key == 'pearson':
            y_lim_top = -1

        plt.rcParams.update({'font.size': 20})
        #plt.figure().set_figwidth(4.5)
        #plt.figure(figsize=(3.75, 4.5))
        props = {'linewidth':2.25}
        # No outliers! whis=(0, 100)
        plt.boxplot(data_to_plot, whis=(0, 100), widths=.85,
                    boxprops=props, flierprops=props, medianprops=props, capprops=props, whiskerprops=props)

        #plt.xlabel(corrupt+' -- '+key)
        plt.ylim(y_lim_bottom, y_lim_top)

        plt.xticks(data_ticks, [model_pretty_name(lbl) for lbl in data_tick_labels], rotation='vertical')
        # Pad margins so that markers don't get clipped by the axes
        plt.margins(0.2)
        # Tweak spacing to prevent clipping of tick-labels
        plt.subplots_adjust(bottom=0.15)

        plt.savefig(folder+fig_folder+corrupt+'_'+key+'.pdf', format="pdf", bbox_inches="tight")
        #plt.show()
        plt.close()

    # Get numbers for each model, regardless of dataset.
    #for key in ['r2', 'pearson', 'brier', 'BA']:
    #    plot_results('LCWA', key, best_calib_models)

    for corrupt in ['LCWA', 'TCLCWA', 'Global', 'Local']:
        for key in ['r2', 'pearson', 'brier', 'BA']:
            plot_results(corrupt, key, calib_models_other_strategies[corrupt].values())

    # Global and local merged.
    for key in ['r2', 'pearson', 'brier', 'BA']:
        merged = list(calib_models_other_strategies['Global'].values())
        merged += list(calib_models_other_strategies['Local'].values())

        plot_results('GlobalLocal', key, calib_models_other_strategies[corrupt].values())

    # TODO Get best technique (isotonic/platt) and targets.
    #print(best_calib_models)

    # For each dataset, sort models by mean rank and by positive probability.
    for dataset in [1, 2, 3, 5, 6, 7, 8, 9]:
        print('Dataset:', dataset)
        calib_model_lst = []
        for calib_model_data in best_calib_models.values():
            if calib_model_data['dataset'] == str(dataset):
                calib_model_lst.append(calib_model_data)

        all_metrics, all_models = ['mr', 'mp', 'TPR'], set()
        partial_ranks_by_metric = {}
        for key in all_metrics:
            calib_model_lst = sorted(calib_model_lst, key=lambda x: float(x[key]), reverse=(key=='mp' or key=='TPR'))

            def scale_mr(mr_value):
                if key == 'mr':
                    min_mr, max_mr = float(calib_model_lst[0][key]), float(calib_model_lst[len(calib_model_lst)-1][key])
                    return 1 - (mr_value - min_mr)/(max_mr - min_mr)
                else:
                    return mr_value

            print('\tRank for ', key, ': ', [model_pretty_name(value['model'], long_output=False)+
                                             ' ({:.2f}'.format(scale_mr(float(value[key])))+')' for value in calib_model_lst])

            # Get partial ranks: [a, b, c] => [a>b, a>c, b>c]
            partial_ranks = set()
            for i in range(len(calib_model_lst)):
                i_model = calib_model_lst[i]['model']
                all_models.add(i_model)
                for j in range(i+1, len(calib_model_lst)):
                    j_model = calib_model_lst[j]['model']
                    partial_ranks.add(i_model+'>'+j_model)
                    all_models.add(j_model)

            partial_ranks_by_metric[key] = partial_ranks

        # Check discrepancies: two models that were ranked differently in relative terms.
        for i in range(len(all_metrics)):
            for j in range(i+1, len(all_metrics)):
                i_metric, j_metric = all_metrics[i], all_metrics[j]
                i_partial_ranks, j_partial_ranks = partial_ranks_by_metric[i_metric], partial_ranks_by_metric[j_metric]

                def overlap(s1, s2, total):
                    return float(len(set(s1).intersection(s2))) / (total*(total-1)/2)

                print('Metric', i_metric, ' vs. ', j_metric, '; Overlap:',
                      overlap(i_partial_ranks, j_partial_ranks, len(all_models)))


if __name__ == '__main__':
    run()
