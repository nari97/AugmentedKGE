import os

import pandas as pd
from matplotlib import pyplot as plt


def extract_from_logs():
    logs_folder = "D:\PhD\Work\AugmentedKGE\AugmentedKGE\AugmentedKGE\Logs"
    table = {}
    for filename in os.listdir(logs_folder):

        file_path = os.path.join(logs_folder, filename)

        with open(file_path) as opened_file:
            model_name = ""
            dataset_name = ""

            metrics = []
            adjusted_metrics = []
            ties = []
            for line in opened_file:
                line = line.strip()

                try:
                    if "Model" in line and "Dataset" in line:
                        splits = line.split(";")

                        model_name = splits[0].split(":")[1].strip()
                        dataset_name = splits[1].split(":")[1].strip()

                    if "Adjusted metric" in line:
                        adjusted_metrics.append(float(line.split(":")[1].strip()))

                    elif "Metric" in line:
                        metrics.append(float(line.split(":")[1].strip()))
                    elif "Ties" in line:
                        ties.append(float(line.strip().split(";")[1].strip().split(":")[1]))

                    if model_name != '' and dataset_name != '':

                        if (dataset_name, model_name) not in table:
                            table[(dataset_name, model_name)] = (adjusted_metrics[1], metrics[5], ties[1])
                except:
                    pass

    dataset_names = set()
    model_names = set()

    for dataset, model in table:
        dataset_names.add(dataset)
        model_names.add(model)

    for dataset in dataset_names:
        models = []
        mrs = []
        ms = []
        ties = []
        for model in model_names:

            if (dataset, model) in table:
                models.append(model)
                mrs.append(table[(dataset, model)][0])
                ms.append(table[(dataset, model)][1])
                ties.append(table[(dataset, model)][2])

        df = pd.DataFrame({"Model name": models, "MR": mrs, "|M|": ms, "Ties": ties})

        fig, ax = plt.subplots(figsize=(10, 3))

        # remove axis
        ax.axis('off')

        # add a title to the figure
        fig.suptitle(f'Dataset name: {dataset}')

        # add the dataframe to the axis object as a table
        ax.table(cellText=df.values, colLabels=df.columns, loc='center')

        # save the figure as an image with the specified filename and format
        plt.savefig(f'Tables/{dataset}.png', bbox_inches='tight', pad_inches=0.5, dpi=300)


if __name__ == "__main__":
    extract_from_logs()
