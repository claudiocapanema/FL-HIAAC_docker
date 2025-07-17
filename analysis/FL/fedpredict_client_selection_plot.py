from pathlib import Path
import numpy as np
import pandas as pd
import scipy.stats as st

import copy
import sys

from base_plots import bar_plot, line_plot, ecdf_plot
import matplotlib.pyplot as plt
import seaborn as sns
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
                df["Dataset"] = np.array([dataset.replace("CIFAR10", "CIFAR-10")] * len(df))
                df["Table"] = np.array([solution_strategy_version[solution]["Table"]] * len(df))
                df["Selection type"] = np.array([solution_strategy_version[solution]["Selection type"]] * len(df))
                df["Strategy"] = np.array([solution_strategy_version[solution]["Strategy"]] * len(df))
                df["Version"] = np.array([solution_strategy_version[solution]["Version"]] * len(df))
                # print("fracao2: ", len(df["Fraction fit"]), len(df))
                df["Selection level"] = np.array([selection_level[float(df["Fraction fit"].iloc[0])]] * len(df))

                if df_concat is None:
                    df_concat = df
                elif len(df) > 0:
                    df_concat = pd.concat([df_concat, df])

                strategy = solution_strategy_version[solution]["Strategy"]
                if strategy not in hue_order:
                    hue_order.append(strategy)
            except Exception as e:
                print("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))

    return df_concat, hue_order


def joint_plot_acc_four_plots(df_test, dataset, write_dir):

        alphas = df_test["Alpha"].unique().tolist()
        datast = df_test['Dataset'].unique().tolist()
        selection_types = df_test['Selection type'].unique().tolist()
        selection_levels = df_test['Selection level'].unique().tolist()
        # exit()
        # figsize=(12, 9),
        sns.set(style='whitegrid')
        rows = len(alphas)
        cols = len(selection_types)
        fig, axs = plt.subplots(rows, cols,  sharex='all', sharey='all', figsize=(9, 6))

        x_column = 'Round (t)'
        y_column = 'Accuracy (%)'
        plt.xlabel(x_column)
        plt.ylabel(y_column)
        # ====================================================================
        for i in range(rows):
            for j in range(cols):
                alpha = alphas[i]
                # dataset = datast[j]
                # dataset = 'GTSRB'
                # dataset = 'CIFAR-10'
                client_selection = ['Random', 'POC', 'RAWCS'][j]
                # client_selection = 'RAWCS'
                fraction_fit = selection_levels[j]
                print("cf: ", client_selection, fraction_fit)
                title = """\u03B1={}; {}""".format(alpha, client_selection)
                filename = ''
                hue_order = ['FedAvg+FP', 'FedAvg']
                hue = "Table"
                # hue_order = ['FedAvg', 'CDA-FedAvg', 'FedCDM', 'FedPer']
                # hue_order = None
                style = 'Selection level'
                # "+FP",
                # style_order = [r"+FP$_{DYN}$",  "+FP", "Original"]
                style_order = ['High', 'Medium', 'Low']
                y_max = 100
                # markers = [',', '.'
                markers = None
                size = None
                # sizes = (1, 1.8)
                sizes = None
                df = df_test[df_test['Dataset'] == dataset]
                df = df[df["Alpha"] == alpha]
                df = df[df["Selection type"] == selection_types[j]]
                # df = df[df["Selection level"] == selection_levels[j]]
                line_plot(df=df, base_dir=write_dir, file_name=filename, x_column=x_column, y_column=y_column,
                          title=title, hue=hue, ax=axs[i, j], tipo='', hue_order=hue_order, style=style,
                          markers=markers, size=size, sizes=sizes, y_max=y_max, y_lim=True, style_order=style_order)
                # if i != 1 and j != 0:
                #     axs[i,j].get_legend().remove()
                #     axs[i,j].legend(fontsize=7)

                axs[i,j].set_xlabel('')
                axs[i,j].set_ylabel('')

        axs[0, 0].get_legend().remove()
        axs[0, 1].get_legend().remove()
        axs[1, 0].get_legend().remove()
        axs[1, 1].get_legend().remove()
        axs[1, 2].get_legend().remove()
        axs[0, 2].get_legend().remove()

        # # =========================///////////================================
        fig.suptitle(dataset, fontsize=16)
        plt.tight_layout()
        plt.subplots_adjust(wspace=0.07, hspace=0.14)
        # plt.subplots_adjust(right=0.9)
        # fig.legend(
        #            loc="Lower right")
        # fig.legend(lines, labels)
        # plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
        fig.supxlabel(x_column, y=-0.02)
        fig.supylabel(y_column, x=-0.01)

        lines_labels = [axs[0, 0].get_legend_handles_labels()]
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
        markers = ["", "-", "--", "dotted"]

        f = lambda m, c: plt.plot([], [], marker=m, color=c, ls="none")[0]
        handles = [f("o", colors[i]) for i in range(len(hue_order) + 1)]
        handles += [plt.Line2D([], [], linestyle=markers[i], color="k") for i in range(len(markers))]
        axs[1, 1].legend(handles, labels, fontsize=7)
        figure = fig.get_figure()
        # print(handles)
        # print("---")
        # print(labels)
        Path(write_dir + "png/").mkdir(parents=True, exist_ok=True)
        Path(write_dir + "svg/").mkdir(parents=True, exist_ok=True)
        filename = f"client_selection_{dataset}"
        print(write_dir + "png/" + filename + ".png")
        figure.savefig(write_dir + "png/" + filename + ".png", bbox_inches='tight', dpi=400)
        figure.savefig(write_dir + "svg/" + filename + ".svg", bbox_inches='tight', dpi=400)

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
    # print(df)
    # print(df["Solution"].unique())
    # exit()

    # table(df, write_path, "Balanced accuracy (%)", t=None)
    joint_plot_acc_four_plots(df, "CIFAR-10", write_path)
    joint_plot_acc_four_plots(df, "CIFAR-10", write_path)
    # table(df, write_path, "Balanced accuracy (%)", t=100)
    # table(df, write_path, "Accuracy (%)", t=100)