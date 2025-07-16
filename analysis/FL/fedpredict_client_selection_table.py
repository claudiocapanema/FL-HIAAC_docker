from pathlib import Path
import numpy as np
import pandas as pd
import scipy.stats as st

import copy
import sys

from base_plots import bar_plot, line_plot, ecdf_plot
import matplotlib.pyplot as plt

def read_data(read_solutions, read_dataset_order):

    df_concat = None
    solution_strategy_version = {
        "FedAvg+FP": {"Strategy": "FedAvg", "Version": "FP", "Table": "FedAvg+FP", "Selection type": "Random"},
        "FedAvg": {"Strategy": "FedAvg", "Version": "Original", "Table": "FedAvg", "Selection type": "Random"},
        "FedAvgRAWCS": {"Strategy": "FedAvg", "Version": "Original", "Table": "FedAvg", "Selection type": "RAWCS"},
        "FedAvgPOC+FP": {"Strategy": "FedAvg", "Version": "FP", "Table": "FedAvg+FP",
                      "Selection type": "POC"},
        "FedAvgRAWCS+FP": {"Strategy": "FedAvg", "Version": "FP", "Table": "FedAvg+FP",
                         "Selection type": "RAWCS"},
        "FedAvgPOC": {"Strategy": "FedAvg", "Version": "Original", "Table": "FedAvg", "Selection type": "POC"},
        "FedYogi+FP": {"Strategy": "FedYogi", "Version": "FP", "Table": "FedYogi+FP", "Selection type": "Random"},
        "FedYogi": {"Strategy": "FedYogi", "Version": "Original", "Table": "FedYogi", "Selection type": "Random"},
        "FedPer": {"Strategy": "FedPer", "Version": "Original", "Table": "FedPer", "Selection type": "Random"},
        "FedKD": {"Strategy": "FedKD", "Version": "Original", "Table": "FedKD", "Selection type": "Random"},
        "FedKD+FP": {"Strategy": "FedKD", "Version": "FP", "Table": "FedKD+FP", "Selection type": "Random"}
    }
    hue_order = []
    selection_level = {0.3: "Low", 0.5: "Medium", 0.7: "High"}
    for solution in read_solutions:

        paths = read_solutions[solution]
        for i in range(len(paths)):
            try:
                dataset = read_dataset_order[i]
                path = paths[i]
                df = pd.read_csv(path)
                df["Solution"] = np.array([solution] * len(df))
                df["Accuracy (%)"] = df["Accuracy"] * 100
                df["Balanced accuracy (%)"] = df["Balanced accuracy"] * 100
                df["Dataset"] = np.array([dataset] * len(df))
                df["Table"] = np.array([solution_strategy_version[solution]["Table"]] * len(df))
                df["Selection type"] = np.array([solution_strategy_version[solution]["Selection type"]] * len(df))
                df["Strategy"] = np.array([solution_strategy_version[solution]["Strategy"]] * len(df))
                df["Version"] = np.array([solution_strategy_version[solution]["Version"]] * len(df))
                df["Selection level"] = np.array([selection_level[float(df["Fraction fit"].iloc[0])]] * len(df))

                if df_concat is None:
                    df_concat = df
                else:
                    df_concat = pd.concat([df_concat, df])

                strategy = solution_strategy_version[solution]["Strategy"]
                if strategy not in hue_order:
                    hue_order.append(strategy)
            except Exception as e:
                print("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))

    return df_concat, hue_order


def table(df, write_path, metric, dataset, t=None):
    df = df[df["Dataset"] == dataset]
    selection_types = df["Selection type"].unique().tolist()
    alphas = df["Alpha"].unique().tolist()
    columns = df["Table"].unique().tolist()
    selection_levels = df["Selection level"].unique().tolist()

    n_strategies = str(len(columns))

    print(columns)

    model_report = {i: {} for i in alphas}
    if t is not None:
        df = df[df['Round (t)'] == t]

    df_test = df[
        ['Round (t)', 'Table', 'Balanced accuracy (%)', 'Accuracy (%)', 'Fraction fit', 'Dataset',
         'Alpha', 'Selection level', 'Selection type']]

    # df_test = df_test.query("""Round in [10, 100]""")
    print("agrupou table")
    experiment = 1
    print(df_test)

    # arr = []
    # for selection_type in selection_types:
    #     arr += [selection_type] * len(columns)
    # index = [np.array(arr),
    #          np.array(columns * len(selection_types))]
    index = [np.array(['High'] * len(columns) * len(selection_types) + ['Medium'] * len(columns) * len(selection_types) + ['Low'] * len(columns) * len(selection_types)), np.array(['Random'] * len(columns) + ['POC'] * len(columns) + ['Random'] * len(columns) + ['POC'] * len(columns) + ['Random'] * len(columns) + ['POC'] * len(columns)),
             np.array(columns * 3 * len(selection_types))]

    models_dict = {}
    ci = 0.95

    for alpha in model_report:
        models_datasets_dict = {dt: {selection_level: {} for selection_level in selection_levels} for dt in selection_types}
        for column in columns:
            for selection_type in selection_types:
                for selection_level in selection_levels:
                    print(alpha, column, selection_type, selection_level)
                    models_datasets_dict[selection_type][selection_level][column] = t_distribution((filter(df_test, selection_type,
                                                                              alpha=float(alpha), selection_level=selection_level, strategy=column)[
                        metric]).tolist(), ci)

        model_metrics = []

        for selection_type in selection_types:
            for selection_level in selection_levels:
                for column in columns:
                    model_metrics.append(models_datasets_dict[selection_type][selection_level][column])

        models_dict[alpha] = model_metrics

    print(models_dict)
    print(index)

    df_table = pd.DataFrame(models_dict, index=index).round(4)
    print("df table: ", df_table)

    print(df_table.to_string())

    df_accuracy_improvements = accuracy_improvement(df_table, selection_types, selection_levels)

    indexes = df_table.index.tolist()
    n_solutions = len(pd.Series([i[1] for i in indexes]).unique().tolist())
    max_values = idmax(df_table, n_solutions)
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

    indexes = alphas
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
        "\multirow[t]{" + n_strategies + "}{*}{GTSRB}", "GTSRB").replace("\cline{1-4}", "\hline")

    Path(write_path).mkdir(parents=True, exist_ok=True)
    if t is not None:
        filename = """{}latex_round_{}_{}_client_selectipn.txt""".format(write_path, t, metric)
    else:
        filename = """{}latex_{}_client_selection.txt""".format(write_path, metric)
    pd.DataFrame({'latex': [latex]}).to_csv(filename, header=False, index=False)

    improvements(df_table, selection_types, metric, selection_levels)
    print(filename)
    #  df.to_latex().replace("\}", "}").replace("\{", "{").replace("\\\nRecall", "\\\n\hline\nRecall").replace("\\\nF-score", "\\\n\hline\nF1-score")


def improvements(df, datasets, metric, selection_levels):
    # , "FedKD+FP": "FedKD"
    # strategies = {"FedAvg+FP": "FedAvg", "FedYogi+FP": "FedYogi", "FedKD+FP": "FedKD"}
    strategies = {"FedAvg+FP": "FedAvg"}
    # strategies = {r"MultiFedAvg+FP": "MultiFedAvg"}
    columns = df.columns.tolist()
    improvements_dict = {'Dataset': [], 'Table': [], 'Original strategy': [], 'Alpha': [], metric: []}
    df_improvements = pd.DataFrame(improvements_dict)

    for dataset in datasets:
        for strategy in strategies:
            original_strategy = strategies[strategy]
            for selection_level in selection_levels:
                for j in range(len(columns)):
                    index = (selection_level, dataset, strategy)
                    index_original = (selection_level, dataset, original_strategy)
                    print(df)
                    print("indice: ", index)
                    acc = float(df.loc[index].tolist()[j].replace("textbf{", "").replace(u"\u00B1", "")[:4])
                    acc_original = float(
                        df.loc[index_original].tolist()[j].replace("textbf{", "")[:4].replace(u"\u00B1", ""))

                    row = {'Dataset': [dataset], 'Table': [strategy], 'Original strategy': [original_strategy],
                           'Alpha': [columns[j]], metric: [acc - acc_original]}
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


def filter(df, selection_type, alpha, selection_level, strategy=None):
    # df['Balanced accuracy (%)'] = df['Balanced accuracy (%)']*100
    if strategy is not None:
        df = df[df['Selection type'] == selection_type]
        df = df.query(
            """Table=='{}'""".format(strategy))
        df = df[df['Alpha'] == alpha]
        df = df[df['Selection level'] == selection_level]
    else:
        df = df[df['Selection type'] == selection_type]
        df = df[df['Alpha'] == alpha]
        df = df[df['Selection level'] == selection_level]

    print("filtrou: ", df, selection_type, alpha, strategy)

    return df


def t_distribution(data, ci):
    if len(data) > 1:
        min_ = st.t.interval(confidence=ci, df=len(data) - 1,
                             loc=np.mean(data),
                             scale=st.sem(data))[0]

        mean = np.mean(data)
        average_variation = (mean - min_).round(1)
        mean = mean.round(1)

        return str(mean) + u"\u00B1" + str(average_variation)
    else:
        return str(round(data[0], 1)) + u"\u00B1" + str(0.0)


def accuracy_improvement(df, selection_types, selection_levels):
    df_difference = copy.deepcopy(df)
    columns = df.columns.tolist()
    indexes = df.index.tolist()
    solutions = pd.Series([i[1] for i in indexes]).unique().tolist()
    # reference_solutions = {"MultiFedAvg+FP": "MultiFedAvg", "MultiFedYogi+FP": "MultiFedYogi", "FedAvgGlobalModelEval+FP": "FedAvgGlobalModelEval", "MultiFedKD+FP": "FedKD"}
    # reference_solutions = {"MultiFedAvg+FP": "MultiFedAvg", "MultiFedAvgGlobalModelEval+FP": "MultiFedAvgGlobalModelEval"}
    # ,
    #                            "FedKD+FP": "FedKD"
    # reference_solutions = {"FedAvg+FP": "FedAvg", "FedYogi+FP": "FedYogi", "FedKD+FP": "FedKD"}
    reference_solutions = {"FedAvg+FP": "FedAvg"}

    print(df_difference)
    # exit()

    for selection_type in selection_types:
        for selection_level in selection_levels:
            for solution in reference_solutions:
                reference_index = (selection_level, selection_type, solution)
                target_index = (selection_level, selection_type, reference_solutions[solution])

                for column in columns:
                    difference = str(round(float(df.loc[reference_index, column].replace(u"\u00B1", "")[:4]) - float(
                        df.loc[target_index, column].replace(u"\u00B1", "")[:4]), 1))
                    difference = str(
                        round(float(difference) * 100 / float(df.loc[target_index, column][:4].replace(u"\u00B1", "")), 1))
                    if difference[0] != "-":
                        difference = r"\textuparrow" + difference
                    else:
                        difference = r"\textdownarrow" + difference.replace("-", "")
                    df_difference.loc[reference_index, column] = "(" + difference + "\%)" + df.loc[reference_index, column]

    return df_difference


def select_mean(index, column_values, columns, n_solutions):
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
        max_tuple = max(dataset_values, key=lambda e: e[0])
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


def idmax(df, n_solutions):
    df_indexes = []
    columns = df.columns.tolist()
    print("colunas", columns)
    for i in range(len(columns)):
        column = columns[i]
        column_values = df[column].tolist()
        print("ddd", column_values)
        indexes = select_mean(i, column_values, columns, n_solutions)
        df_indexes += indexes

    return df_indexes


if __name__ == "__main__":
    # experiment_id = "1_new_clients"
    experiment_id = "1"
    cd = "false"
    total_clients = 20
    alphas = [0.1, 1.0]
    dataset = ["CIFAR10", "GTSRB"]
    # dataset = ["EMNIST", "CIFAR10"]
    # models_names = ["cnn_c"]
    model_name = "CNN_3"
    fractions_fit = [0.3, 0.5, 0.7]
    number_of_rounds = 100
    local_epochs = 1
    fraction_new_clients = alphas[0]
    round_new_clients = 0
    train_test = "test"
    # solutions = ["MultiFedAvgWithFedPredict", "MultiFedAvg",
    #              "MultiFedAvgGlobalModelEvalWithFedPredict", "MultiFedAvgGlobalModelEval",
    #              "MultiFedYogiWithFedPredict", "MultiFedYogi", "MultiFedYogiGlobalModelEval", "MultiFedPer"]
    # solutions = ["MultiFedAvgWithFedPredict", "MultiFedAvg", "MultiFedAvgGlobalModelEval",
    #              "MultiFedAvgGlobalModelEvalWithFedPredict", "MultiFedPer"]
    solutions = ["FedAvg+FP", "FedAvgPOC+FP", "FedAvg", "FedAvgPOC", "FedAvgRAWCS", "FedAvgRAWCS+FP"]
    # solutions = ["MultiFedAvgWithFedPredict", "MultiFedAvg"]

    read_solutions = {solution: [] for solution in solutions}
    read_dataset_order = []
    for solution in solutions:
        for alpha in alphas:
            for dt in dataset:
                for fraction_fit in fractions_fit:
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
                    solution_file = solution
                    read_solutions[solution].append("""{}{}_{}.csv""".format(read_path, dt, solution_file))

    write_path = """plots/FL/experiment_id_{}/new_clients_fraction_{}_round_{}/clients_{}/alpha_{}/alpha_end_{}_{}/{}/concept_drift_rounds_{}_{}/{}/fc_{}/rounds_{}/epochs_{}/""".format(
        experiment_id,
        fraction_new_clients,
        round_new_clients,
        total_clients,
        [str(alphas)],
        alphas,
        alphas,
        dataset,
        0,
        0,
        model_name,
        fractions_fit,
        number_of_rounds,
        local_epochs)

    print(read_solutions)

    df, hue_order = read_data(read_solutions, read_dataset_order)
    print(df)

    # table(df, write_path, "Balanced accuracy (%)", t=None)
    table(df, write_path, "Accuracy (%)", dataset="CIFAR10", t=None)
    # table(df, write_path, "Balanced accuracy (%)", t=100)
    # table(df, write_path, "Accuracy (%)", t=100)