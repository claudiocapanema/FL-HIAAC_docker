from pathlib import Path
import numpy as np
import pandas as pd
import scipy.stats as st

import copy
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.stats as st

import copy
import sys

from base_plots import bar_plot, line_plot, ecdf_plot
import matplotlib.pyplot as plt
# Alterar opções para exibir tudo
pd.set_option('display.max_rows', None)      # Mostra todas as linhas
pd.set_option('display.max_columns', None)   # Mostra todas as colunas
pd.set_option('display.width', None)         # Não quebra linha horizontal
pd.set_option('display.max_colwidth', None)

REFERENCE_STRATEGY = {"FedAvg+FP": "FedAvg", "FedAvg+FP$_{dc}$": "FedAvg", "FedAvg+FP$_{d}$": "FedAvg", "FedAvg+FP$_{c}$": "FedAvg", "FedAvg+FP$_{s}$": "FedAvg", "FedAvg+FP$_{per}$": "FedAvg", "FedAvg+FP$_{kd}$": "FedAvg", "FedAvg": "FedAvg"}

def parameters_reduction(df):
    model = df["Model"].tolist()[0]
    dataset = df["Dataset"].tolist()[0]
    alpha = df["Alpha"].tolist()[0]

    reference_solution = df.query(f"Model == '{model}' & Dataset == '{dataset}'")["Size (MB)"]
    df["Parameters saving (%)"] = (df["Size (MB)"] - reference_solution) / reference_solution
    print(df)
    # return df

def read_data(read_solutions, read_dataset_order, read_model_order):

    df_concat = None
    df_concat_all = None
    solution_strategy_version = {
        "FedAvg+FP": {"Strategy": "FedAvg", "Version": "FP", "Table": "FedAvg+FP"},
        "FedAvg+FP_dls_compredict": {"Strategy": "FedAvg", "Version": "FP$_{dc}$", "Table": "FedAvg+FP$_{dc}$"},
        "FedAvg+FP_dls": {"Strategy": "FedAvg", "Version": "FP$_{d}$", "Table": "FedAvg+FP$_{d}$"},
        "FedAvg+FP_compredict": {"Strategy": "FedAvg", "Version": "FP$_{c}$", "Table": "FedAvg+FP$_{c}$"},
        "FedAvg+FP_sparsification": {"Strategy": "FedAvg", "Version": "FP$_{s}$", "Table": "FedAvg+FP$_{s}$"},
        "FedAvg+FP_per": {"Strategy": "FedAvg", "Version": "FP$_{per}$", "Table": "FedAvg+FP$_{per}$"},
        "FedAvg+FP_fedkd": {"Strategy": "FedAvg", "Version": "FP$_{kd}$", "Table": "FedAvg+FP$_{kd}$"},
        "FedAvg": {"Strategy": "FedAvg", "Version": "Original", "Table": "FedAvg"},
        "FedYogi+FP": {"Strategy": "FedYogi", "Version": "FP$_{dc}$", "Table": "FedYogi+FP$_{dc}$"},
        "FedYogi": {"Strategy": "FedYogi", "Version": "Original", "Table": "FedYogi"},
        "FedPer": {"Strategy": "FedPer", "Version": "Original", "Table": "FedPer"},
        "FedKD": {"Strategy": "FedKD", "Version": "Original", "Table": "FedKD"},
        "FedKD+FP": {"Strategy": "FedKD", "Version": "FP$_{dc}$", "Table": "FedKD+FP$_{dc}$"}
    }
    hue_order = []
    for solution in read_solutions:

        paths = read_solutions[solution]
        for i in range(len(paths)):
            try:
                dataset = read_dataset_order[i]
                model = read_model_order[i]
                path = paths[i]
                df = pd.read_csv(path)
                df["Solution"] = np.array([solution] * len(df))
                df["Accuracy (%)"] = df["Accuracy"] * 100
                df["Balanced accuracy (%)"] = df["Balanced accuracy"] * 100
                df["Dataset"] = np.array([dataset] * len(df))
                df["Model"] = np.array([model.replace("CNN_2", "CNN-a").replace("CNN_3", "CNN-b")] * len(df))
                df["Table"] = np.array([solution_strategy_version[solution]["Table"]] * len(df))
                df["Strategy"] = np.array([solution_strategy_version[solution]["Strategy"]] * len(df))
                df["Version"] = np.array([solution_strategy_version[solution]["Version"]] * len(df))
                df["Size (MB)"] = (df["Model size"] * 0.000001)
                print(df["Size (MB)"])
                if "+FP_" in solution:
                    df["Size (MB)"] = (df["Model size (compressed)"] * 0.000001)

                if df_concat is None:
                    df_concat = df
                else:
                    df_concat = pd.concat([df_concat, df])

                strategy = solution_strategy_version[solution]["Strategy"]
                if strategy not in hue_order:
                    hue_order.append(strategy)
            except Exception as e:
                print("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))

    models = df_concat["Model"].unique().tolist()
    datasets = df_concat["Dataset"].unique().tolist()
    alphas = df_concat["Alpha"].unique().tolist()
    solutions = df_concat["Table"].unique().tolist()
    for model in models:
        for dataset in datasets:
            for alpha in alphas:
                for solution in solutions:
                    reference_solution = df_concat.query(
                        f"Model == '{model}' & Dataset == '{dataset}' & Alpha == {alpha} & Table == '{REFERENCE_STRATEGY[solution]}'")[
                        "Size (MB)"]
                    df_solution = df_concat.query(f"Model == '{model}' & Dataset == '{dataset}' & Alpha == {alpha} & Table == '{solution}'")
                    df_solution["Parameters saving (%)"] = 100 * ((-df_solution["Size (MB)"] + reference_solution) / reference_solution)
                    if df_concat_all is None:
                        df_concat_all = df_solution
                    else:
                        df_concat_all = pd.concat([df_concat_all, df_solution])

    # df_concat = df_concat.groupby(["Model", "Dataset", "Alpha"]).apply(lambda e: parameters_reduction(e)).reset_index()

    print(df_concat_all[["Table", "Dataset", "Model", "Alpha", "Round (t)", "Size (MB)", "Parameters saving (%)"]])

    return df_concat_all, hue_order


def evaluate_client_joint_parameter_reduction(df, base_dir, alpha=0.1):

    print("base dir: ", base_dir)
    x_column = 'Round (t)'
    y_column = 'Accuracy (%)'
    hue = 'Table'
    # style = '\u03B1'
    style = None
    y_min = 0
    compression = ["FedAvg+FP$_{dc}$", "FedAvg+FP$_{d}$", "FedAvg+FP$_{c}$", "FedAvg+FP$_{kd}$", "FedAvg+FP$_{s}$", "FedAvg+FP$_{per}$", "FedAvg+FP", "FedAvg"]

    y_max = 100

    fig, ax = plt.subplots(3, 2, sharex='all', sharey='all', figsize=(6, 6))
    x = df[x_column].tolist()
    y = df[y_column].tolist()
    datasets = df["Dataset"].unique().tolist()
    models = df["Model"].unique().tolist()

    # df = df.query(f"Dataset == 'CIFAR10' & Model == 'CNN_2' & Alpha == 0.1")

    for i in range(3):
        for j in range(2):
            dataset = datasets[i]
            model = models[j]
            title = f"{dataset}; {model}"
            # if dataset == "CIFAR10" and model == "CNN-a":
            #     df_aux = df.query(f"Dataset == '{dataset}' and Model == '{model}'")
            #     df_aux = df_aux[df_aux["Table"].isin(["FedAvg+FP$_{d}$"])]
            # else:
            #     df_aux = df.query(f"Dataset == '{dataset}' and Model == '{model}'")
            df_aux = df.query(f"Dataset == '{dataset}' and Model == '{model}'")
            line_plot(ax=ax[i, j],
                      df=df_aux,
                      base_dir=base_dir,
                      file_name="parameter_reduction",
                      x_column=x_column,
                      y_column=y_column,
                      title=title,
                      hue=hue,
                      style=style,
                      hue_order=compression,
                      tipo=1,
                      log_scale=False,
                      y_lim=True,
                      y_max=y_max,
                      y_min=y_min,
                      n=1)

            ax[i, j].get_legend().remove()
            ax[i, j].set_xlabel('')
            ax[i, j].set_ylabel('')

    # fig.suptitle("", fontsize=16)
    # fig.supxlabel(x_column, y=-0.02)
    # fig.supylabel(y_column, x=-0.005)
    lines_labels = [ax[1, 0].get_legend_handles_labels()]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    print("linhas")
    print(lines)
    print(lines[0].get_color(), lines[0].get_ls())
    print("rotulos")
    print(labels)
    # # exit()
    colors = []
    markers = []
    for i in range(len(lines)):
        color = lines[i].get_color()
        colors.append(color)
        ls = lines[i].get_ls()
        if ls not in ["o"]:
            ls = "o"
    markers = ["", "-", "--"]

    f = lambda m, c: plt.plot([], [], marker=m, color=c, ls="none")[0]
    handles = [f("o", colors[i]) for i in range(len(colors))]
    fig.legend(handles, labels, loc='upper center', ncol=4, title="""\u03B1={}""".format(alpha),
               bbox_to_anchor=(0.5, 1.06), fontsize=9)

    figure = fig.get_figure()
    Path(base_dir + "png/").mkdir(parents=True, exist_ok=True)
    Path(base_dir + "svg/").mkdir(parents=True, exist_ok=True)
    filename = f"cnn_a_cnn_b_alpha_{alpha}"
    figure.savefig(base_dir + "png/" + filename + ".png", bbox_inches='tight', dpi=400)
    figure.savefig(base_dir + "svg/" + filename + ".svg", bbox_inches='tight', dpi=400)
    print(base_dir + "png/" + filename + ".png")


if __name__ == "__main__":
    # experiment_id = "1_new_clients"
    experiment_id = "1"
    cd = "false"
    total_clients = 20
    alpha = 1.0
    dataset = ["EMNIST", "CIFAR10", "GTSRB"]
    # dataset = ["EMNIST", "CIFAR10"]
    # models_names = ["cnn_c"]
    models_name = ["CNN_2", "CNN_3"]
    fraction_fit = 0.3
    number_of_rounds = 100
    local_epochs = 1
    fraction_new_clients = alpha
    round_new_clients = 0
    train_test = "test"
    # solutions = ["MultiFedAvgWithFedPredict", "MultiFedAvg",
    #              "MultiFedAvgGlobalModelEvalWithFedPredict", "MultiFedAvgGlobalModelEval",
    #              "MultiFedYogiWithFedPredict", "MultiFedYogi", "MultiFedYogiGlobalModelEval", "MultiFedPer"]
    # solutions = ["MultiFedAvgWithFedPredict", "MultiFedAvg", "MultiFedAvgGlobalModelEval",
    #              "MultiFedAvgGlobalModelEvalWithFedPredict", "MultiFedPer"]
    solutions = ["FedAvg", "FedAvg+FP", "FedAvg+FP_compredict", "FedAvg+FP_dls_compredict", "FedAvg+FP_dls", "FedAvg+FP_fedkd", "FedAvg+FP_per", "FedAvg+FP_sparsification"]
    # solutions = ["MultiFedAvgWithFedPredict", "MultiFedAvg"]

    read_solutions = {solution: [] for solution in solutions}
    read_dataset_order = []
    read_model_order = []
    for solution in solutions:
        for model_name in models_name:
            for dt in dataset:
                algo = dt + "_" + solution

                read_path = """/home/gustavo/PycharmProjects/FL-HIAAC_docker/results/experiment_id_{}/clients_{}/alpha_{}/{}/{}/fc_{}/rounds_{}/epochs_{}/{}/""".format(
                    experiment_id,
                    total_clients,
                    alpha,
                    dt,
                    model_name,
                    fraction_fit,
                    number_of_rounds,
                    local_epochs,
                    "test")
                read_dataset_order.append(dt)
                read_model_order.append(model_name)
                solution_file = solution
                read_solutions[solution].append("""{}{}_{}.csv""".format(read_path, dt, solution_file))

    write_path = """plots/FL/experiment_id_{}/new_clients_fraction_{}_round_{}/clients_{}/alpha_{}/alpha_end_{}_{}/{}/concept_drift_rounds_{}_{}/{}/fc_{}/rounds_{}/epochs_{}/""".format(
        experiment_id,
        fraction_new_clients,
        round_new_clients,
        total_clients,
        [str(alpha)],
        alpha,
        alpha,
        dataset,
        0,
        0,
        models_name,
        fraction_fit,
        number_of_rounds,
        local_epochs)

    print(read_solutions)

    df, hue_order = read_data(read_solutions, read_dataset_order, read_model_order)
    print(df)

    evaluate_client_joint_parameter_reduction(df, write_path, alpha=alpha)
    evaluate_client_joint_parameter_reduction(df, write_path, alpha=alpha)