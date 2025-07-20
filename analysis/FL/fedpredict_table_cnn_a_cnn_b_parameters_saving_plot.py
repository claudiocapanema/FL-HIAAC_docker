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
                df["Dataset"] = np.array([dataset.replace("CIFAR10", "CIFAR-10")] * len(df))
                df["Model"] = np.array([model.replace("CNN_2", "CNN-a").replace("CNN_3", "CNN-b")] * len(df))
                df["Table"] = np.array([solution_strategy_version[solution]["Table"]] * len(df))
                df["Strategy"] = np.array([solution_strategy_version[solution]["Strategy"]] * len(df))
                df["Version"] = np.array([solution_strategy_version[solution]["Version"]] * len(df))
                df["Size (MB)"] = (df["Model size"] * 0.000001)
                print(df["Size (MB)"])
                if "+FP_" in solution:
                    df["Model size (compressed)"] = (df["Model size (compressed)"] * 0.000001)
                    df["Parameters saving (%)"] = 100 * ((df["Size (MB)"] - df["Model size (compressed)"]) / df["Size (MB)"])
                    df["Size (MB)"] = df["Model size (compressed)"]
                else:
                    df["Parameters saving (%)"] = np.array([0] * len(df))

                if df_concat is None:
                    df_concat = df
                else:
                    df_concat = pd.concat([df_concat, df])

                strategy = solution_strategy_version[solution]["Strategy"]
                if strategy not in hue_order:
                    hue_order.append(strategy)
            except Exception as e:
                print("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))

    # models = df_concat["Model"].unique().tolist()
    # datasets = df_concat["Dataset"].unique().tolist()
    # alphas = df_concat["Alpha"].unique().tolist()
    # solutions = df_concat["Table"].unique().tolist()
    # for model in models:
    #     for dataset in datasets:
    #         for alpha in alphas:
    #             for solution in solutions:
    #                 reference_solution = df_concat.query(
    #                     f"Model == '{model}' & Dataset == '{dataset}' & Alpha == {alpha} & Table == '{REFERENCE_STRATEGY[solution]}'")[
    #                     "Size (MB)"]
    #                 df_solution = df_concat.query(f"Model == '{model}' & Dataset == '{dataset}' & Alpha == {alpha} & Table == '{solution}'")
    #                 df_solution["Parameters saving (%)"] = 100 * ((-df_solution["Size (MB)"] + reference_solution) / reference_solution)
    #                 if df_concat_all is None:
    #                     df_concat_all = df_solution
    #                 else:
    #                     df_concat_all = pd.concat([df_concat_all, df_solution])

    # df_concat = df_concat.groupby(["Model", "Dataset", "Alpha"]).apply(lambda e: parameters_reduction(e)).reset_index()

    print(df_concat[["Table", "Dataset", "Model", "Alpha", "Round (t)", "Size (MB)", "Parameters saving (%)"]])

    return df_concat, hue_order


def table(df, write_path, metric, t=None, inverse=False):
    datasets = df["Dataset"].unique().tolist()
    models = df["Model"].unique().tolist()
    columns = df["Table"].unique().tolist()
    n_strategies = str(len(columns))

    print(columns)

    model_report = {i: {} for i in models}
    if t is not None:
        df = df[df['Round (t)'] == t]

    df_test = df[
        ['Round (t)', 'Table', 'Balanced accuracy (%)', 'Accuracy (%)', 'Fraction fit', 'Dataset',
         'Alpha', 'Model', 'Size (MB)']]

    # df_test = df_test.query("""Round in [10, 100]""")
    print("agrupou table")
    experiment = 1
    print(df_test)

    arr = []
    for dt in datasets:
        arr += [dt] * len(columns)
    index = [np.array(arr),
             np.array(columns * len(datasets))]

    models_dict = {}
    ci = 0.95

    for alpha in model_report:
        models_datasets_dict = {dt: {} for dt in datasets}
        for column in columns:
            for dt in datasets:

                print("filtro: ", alpha, column, dt)
                models_datasets_dict[dt][column] = t_distribution((filter(df_test, dt,
                                                                          model=alpha, strategy=column)[
                    metric]).tolist(), ci)

        model_metrics = []

        for dt in datasets:
            for column in columns:
                model_metrics.append(models_datasets_dict[dt][column])

        models_dict[alpha] = model_metrics

    print(models_dict)
    print(index)

    df_table = pd.DataFrame(models_dict, index=index).round(4)
    print("df table: ", df_table)

    print(df_table.to_string())

    df_accuracy_improvements = accuracy_improvement(df_table, datasets)

    indexes = df_table.index.tolist()
    n_solutions = len(pd.Series([i[1] for i in indexes]).unique().tolist())
    max_values = idmax(df_table, n_solutions, inverse)
    print("max values", max_values)

    for max_value in max_values:
        row_index = max_value[0]
        column = max_value[1]
        column_values = df_accuracy_improvements[column].tolist()
        column_values[row_index] = "textbf{" + str(column_values[row_index]) + "}"

        df_accuracy_improvements[column] = np.array(column_values)

    df_accuracy_improvements.columns = np.array(list(model_report.keys()))
    print("melhorias")
    print(df_accuracy_improvements)

    indexes = models
    for i in range(df_accuracy_improvements.shape[0]):
        row = df_accuracy_improvements.iloc[i]
        for index in indexes:
            value_string = row[index]
            add_textbf = False
            if "textbf{" in value_string:
                value_string = value_string.replace("textbf{", "").replace("}", "")
                add_textbf = True

            if ")" in value_string:
                value_string = value_string.replace("(", "").split(")")
                gain = value_string[0]
                acc = value_string[1]
            else:
                gain = ""
                acc = value_string

            if add_textbf:
                if gain != "":
                    gain = "textbf{" + gain + "}"
                acc = "textbf{" + acc + "}"

            row[index] = acc + " & " + gain

        df_accuracy_improvements.iloc[i] = row

    latex = df_accuracy_improvements.to_latex().replace("\\\nEMNIST", "\\\n\hline\nEMNIST").replace("\\\nGTSRB",
                                                                                                    "\\\n\hline\nGTSRB").replace(
        "\\\nCIFAR-10", "\\\n\hline\nCIFAR-10").replace("\\bottomrule", "\\hline\n\\bottomrule").replace("\\midrule",
                                                                                                         "\\hline\n\\midrule").replace(
        "\\toprule", "\\hline\n\\toprule").replace("textbf", r"\textbf").replace("\}", "}").replace("\{", "{").replace(
        "\\begin{tabular", "\\resizebox{\columnwidth}{!}{\\begin{tabular}").replace("\$", "$").replace("\&",
                                                                                                       "&").replace(
        "&  &", "& - &").replace("\_", "_").replace(
        "&  \\", "& - \\").replace(" - " + r"\textbf", " " + r"\textbf").replace("_{dc}", r"_{\text{dc}}").replace(
        "\multirow[t]{" + n_strategies + "}{*}{EMNIST}", "EMNIST").replace(
        "\multirow[t]{" + n_strategies + "}{*}{CIFAR10}", "CIFAR10").replace(
        "\multirow[t]{" + n_strategies + "}{*}{GTSRB}", "GTSRB").replace("\cline{1-4}", "\hline").replace("CIFAR10", "CIFAR-10").replace(r"\textuparrow0.0\%", "-")

    Path(write_path).mkdir(parents=True, exist_ok=True)
    if t is not None:
        filename = """{}latex_round_{}_{}.txt""".format(write_path, t, metric)
    else:
        filename = """{}latex_{}.txt""".format(write_path, metric)
    pd.DataFrame({'latex': [latex]}).to_csv(filename, header=False, index=False)

    improvements(df_table, datasets, metric)

    #  df.to_latex().replace("\}", "}").replace("\{", "{").replace("\\\nRecall", "\\\n\hline\nRecall").replace("\\\nF-score", "\\\n\hline\nF1-score")


def improvements(df, datasets, metric):
    # , "FedKD+FP": "FedKD"
    # strategies = {"FedAvg+FP": "FedAvg", "FedYogi+FP": "FedYogi", "FedKD+FP": "FedKD"}
    # strategies = {r"MultiFedAvg+FP": "MultiFedAvg"}
    columns = df.columns.tolist()
    improvements_dict = {'Dataset': [], 'Table': [], 'Original strategy': [], 'Model': [], metric: []}
    df_improvements = pd.DataFrame(improvements_dict)

    for dataset in datasets:
        for strategy in REFERENCE_STRATEGY:
            original_strategy = REFERENCE_STRATEGY[strategy]

            for j in range(len(columns)):
                index = (dataset, strategy)
                index_original = (dataset, original_strategy)
                print(df)
                print("indice: ", index)
                acc = float(df.loc[index].tolist()[j].replace("textbf{", "").replace(u"\u00B1", "")[:4])
                acc_original = float(
                    df.loc[index_original].tolist()[j].replace("textbf{", "")[:4].replace(u"\u00B1", ""))

                row = {'Dataset': [dataset], 'Table': [strategy], 'Original strategy': [original_strategy],
                       'Model': [columns[j]], metric: [acc - acc_original]}
                row = pd.DataFrame(row)

                print(row)

                if len(df_improvements) == 0:
                    df_improvements = row
                else:
                    df_improvements = pd.concat([df_improvements, row], ignore_index=True)

    print(df_improvements)


def groupb_by_plot(self, df, metric):
    accuracy = float(df[metric].mean())
    loss = float(df['Loss'].mean())

    return pd.DataFrame({metric: [accuracy], 'Loss': [loss]})


def filter(df, dataset, model, strategy=None):
    # df['Balanced accuracy (%)'] = df['Balanced accuracy (%)']*100
    if strategy is not None:
        df = df.query(
            """ Dataset=='{}' and Table=='{}'""".format(str(dataset), strategy))
        df = df[df['Model'] == model]
    else:
        df = df.query(
            """and Dataset=='{}'""".format((dataset)))
        df = df[df['Model'] == model]

    print("filtrou: ", df, dataset, model, strategy)

    return df


def t_distribution(data, ci):
    if len(data) > 1:
        min_ = st.t.interval(confidence=ci, df=len(data) - 1,
                             loc=np.mean(data),
                             scale=st.sem(data))[0]

        mean = np.mean(data)
        average_variation = (mean - min_).round(1)
        mean = mean.round(1)
        if np.isnan(average_variation):
            average_variation = 0.0
        return str(mean) + u"\u00B1" + str(average_variation)
    else:
        return str(round(data[0], 1)) + u"\u00B1" + str(0.0)


def accuracy_improvement(df, datasets):
    df_difference = copy.deepcopy(df)
    columns = df.columns.tolist()
    indexes = df.index.tolist()
    solutions = pd.Series([i[1] for i in indexes]).unique().tolist()
    # reference_solutions = {"MultiFedAvg+FP": "MultiFedAvg", "MultiFedYogi+FP": "MultiFedYogi", "FedAvgGlobalModelEval+FP": "FedAvgGlobalModelEval", "MultiFedKD+FP": "FedKD"}
    # reference_solutions = {"MultiFedAvg+FP": "MultiFedAvg", "MultiFedAvgGlobalModelEval+FP": "MultiFedAvgGlobalModelEval"}
    # ,
    #                            "FedKD+FP": "FedKD"
    # reference_solutions = {"FedAvg+FP": "FedAvg", "FedYogi+FP": "FedYogi", "FedKD+FP": "FedKD"}
    reference_solutions = {"FedAvg+FP": "FedAvg", "FedAvg+FP$_{dc}$": "FedAvg", "FedAvg+FP$_{d}$": "FedAvg", "FedAvg+FP$_{c}$": "FedAvg", "FedAvg+FP$_{s}$": "FedAvg", "FedAvg+FP$_{per}$": "FedAvg", "FedAvg+FP$_{kd}$": "FedAvg"}

    print(df_difference)
    # exit()

    for dataset in datasets:
        for solution in reference_solutions:
            reference_index = (dataset, solution)
            target_index = (dataset, reference_solutions[solution])

            for column in columns:
                difference = str(round(float(df.loc[reference_index, column].replace(u"\u00B1", "")[:4]) - float(
                    df.loc[target_index, column].replace(u"\u00B1", "")[:4]), 1))
                print("//: ", float(df.loc[target_index, column][:4].replace(u"\u00B1", "")), target_index, column)
                difference = str(
                    round(float(difference) * 100 / float(df.loc[target_index, column][:4].replace(u"\u00B1", "")), 1))
                if difference[0] != "-":
                    difference = r"\textuparrow" + difference
                else:
                    difference = r"\textdownarrow" + difference.replace("-", "")
                df_difference.loc[reference_index, column] = "(" + difference + "\%)" + df.loc[reference_index, column]

    return df_difference


def select_mean(index, column_values, columns, n_solutions, inverse=False):
    list_of_means = []
    indexes = []
    print("ola: ", column_values, "ola0")

    for i in range(len(column_values)):
        print("valor: ", column_values[i])
        value = float(str(str(column_values[i])[:4]).replace(u"\u00B1", ""))
        interval = float(str(column_values[i])[5:8])
        minimum = value - interval
        maximum = value + interval
        list_of_means.append((value, minimum, maximum))

    for i in range(0, len(list_of_means), n_solutions):

        dataset_values = list_of_means[i: i + n_solutions]
        if not inverse:
            max_tuple = max(dataset_values, key=lambda e: e[0])
        else:
            max_tuple = min(dataset_values, key=lambda e: e[0])
        column_min_value = max_tuple[1]
        column_max_value = max_tuple[2]
        print("maximo: ", column_max_value)
        for j in range(len(list_of_means)):
            value_tuple = list_of_means[j]
            min_value = value_tuple[1]
            max_value = value_tuple[2]
            if j >= i and j < i + n_solutions:
                if not (max_value < column_min_value or min_value > column_max_value):
                    indexes.append([j, columns[index]])

    return indexes


def idmax(df, n_solutions, inverse=False):
    df_indexes = []
    columns = df.columns.tolist()
    print("colunas", columns)
    for i in range(len(columns)):
        column = columns[i]
        column_values = df[column].tolist()
        print("ddd", column_values)
        indexes = select_mean(i, column_values, columns, n_solutions, inverse)
        df_indexes += indexes

    return df_indexes

def evaluate_client_joint_parameter_reduction(df, base_dir):

    print("base dir: ", base_dir)
    x_column = 'Round (t)'
    y_column = 'Parameters saving (%)'
    hue = 'Table'
    # style = '\u03B1'
    style = "Alpha"
    y_min = 0
    compression = ["FedAvg+FP$_{dc}$", "FedAvg+FP$_{d}$", "FedAvg+FP$_{c}$", "FedAvg+FP$_{kd}$", "FedAvg+FP$_{s}$", "FedAvg+FP$_{per}$"]

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
        lines_labels = [ax[0, 1].get_legend_handles_labels()]
        lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
        print("linhas")
        print(lines)
        print(lines[0].get_color(), lines[0].get_ls())
        print("rotulos")
        print(labels)
        colors = []
        for i in range(len(lines)):
            color = lines[i].get_color()
            colors.append(color)
            ls = lines[i].get_ls()
            if ls not in ["o"]:
                ls = "o"
        markers = ["-", "--"]

        f = lambda m, c: plt.plot([], [], marker=m, color=c, ls="none")[0]
        n = len(compression) + 1
        handles = [f("o", colors[i]) for i in range(n)]
        new_labels = []
        for i in range(len(labels)):
            if i != n:
                new_labels.append(labels[i])
            else:
                print("label: ", labels[i])
        new_labels[-1] = '\u03B1=' + new_labels[-1]
        new_labels[-2] = '\u03B1=' + new_labels[-2]
        new_labels = new_labels[1:]

        handles += [plt.Line2D([], [], linestyle=markers[i], color="k") for i in range(len(markers))]
        fig.legend(handles[1:], new_labels, fontsize=9, ncols=4, bbox_to_anchor=(0.91, 1.02))
        figure = fig.get_figure()
        Path(base_dir + "png/").mkdir(parents=True, exist_ok=True)
        Path(base_dir + "svg/").mkdir(parents=True, exist_ok=True)
        filename = f"parameters_reduction_percentage_alpha_{alpha}"
        figure.savefig(base_dir + "png/" + filename + ".png", bbox_inches='tight', dpi=400)
        figure.savefig(base_dir + "svg/" + filename + ".svg", bbox_inches='tight', dpi=400)


if __name__ == "__main__":
    # experiment_id = "1_new_clients"
    experiment_id = "1"
    cd = "false"
    total_clients = 20
    alphas = [0.1, 1.0]
    dataset = ["EMNIST", "CIFAR10", "GTSRB"]
    # dataset = ["EMNIST", "CIFAR10"]
    # models_names = ["cnn_c"]
    models_name = ["CNN_2", "CNN_3"]
    fraction_fit = 0.3
    number_of_rounds = 100
    local_epochs = 1
    fraction_new_clients = 0
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
    read_alpha_order = []
    for solution in solutions:
        for model_name in models_name:
            for dt in dataset:
                for alpha in alphas:
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
                    read_alpha_order.append(alpha)
                    solution_file = solution
                    read_solutions[solution].append("""{}{}_{}.csv""".format(read_path, dt, solution_file))

    write_path = """plots/FL/experiment_id_{}/new_clients_fraction_{}_round_{}/clients_{}/alpha_{}/{}/concept_drift_rounds_{}_{}/{}/fc_{}/rounds_{}/epochs_{}/""".format(
        experiment_id,
        fraction_new_clients,
        round_new_clients,
        total_clients,
        alphas,
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

    # table(df, write_path, "Accuracy (%)", t=None)
    # table(df, write_path, "Accuracy (%)", t=100)
    # table(df, write_path, "Size (MB)", t=None, inverse=True)
    # table(df, write_path, "Size (MB)", t=100, inverse=True)

    evaluate_client_joint_parameter_reduction(df, write_path)
    evaluate_client_joint_parameter_reduction(df, write_path)
    print(write_path)