from pathlib import Path
import numpy as np
import pandas as pd
import sys

from base_plots import line_plot
import ast
from base_plots import bar_plot, line_plot, ecdf_plot, box_plot
import matplotlib.pyplot as plt
import seaborn as sns

def read_data(read_solutions, read_dataset_order):

    df_concat = None
    solution_strategy_version = {
        "FedAvg+FP": {"Strategy": "FedAvg", "Version": "FP$_{dc}$", "Table": "FedAvg+FP"},
        "FedAvg": {"Strategy": "FedAvg", "Version": "Original", "Table": "FedAvg"},
        "FedYogi+FP": {"Strategy": "FedYogi", "Version": "FP$_{dc}$", "Table": "FedYogi+FP"},
        "FedYogi": {"Strategy": "FedYogi", "Version": "Original", "Table": "FedYogi"},
        "FedPer": {"Strategy": "FedPer", "Version": "Original", "Table": "FedPer"},
        "FedKD": {"Strategy": "FedKD", "Version": "Original", "Table": "FedKD"},
        "FedKD+FP": {"Strategy": "FedKD", "Version": "FP$_{dc}$", "Table": "FedKD+FP"}
    }
    hue_order = []
    for solution in read_solutions:

        paths = read_solutions[solution]
        for i in range(len(paths)):
            try:
                dataset = read_dataset_order[i]
                path = paths[i]
                nts = []
                rounds = []
                accuracies = []
                df = pd.read_csv(path)
                for i in range(len(df)):
                    row = df.iloc[i]
                    nts += ast.literal_eval(row["nt"])
                    rounds += [row["Round (t)"]] * len(ast.literal_eval(row["nt"]))
                    accuracies += ast.literal_eval(row["Accuracy (%)"])

                accuracies = np.array(accuracies) * 100
                df = pd.DataFrame({"Accuracy (%)": accuracies, "Round (t)": rounds, "nt": nts})
                df["Solution"] = np.array([solution] * len(accuracies))
                df["Dataset"] = np.array([dataset] * len(accuracies))
                df["Strategy"] = np.array([solution_strategy_version[solution]["Strategy"]] * len(accuracies))
                df["Version"] = np.array([solution_strategy_version[solution]["Version"]] * len(accuracies))

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

def filter(df, dataset, strategy=None):

    # df['Accuracy (%)'] = df['Accuracy (%)']*100
    if strategy is not None:
        df = df.query(
            """Dataset=='{}' and Strategy=='{}'""".format(str(dataset), strategy))
    else:
        df = df.query(
            """Dataset=='{}'""".format(dataset))

    print("filtrou: ", df)

    return df

def filter_and_plot(ax, base_dir, filename, title, df, dataset, x_column, y_column, hue, hue_order=None, style=None, x_order=None, palette=None):

    df = filter(df, dataset)

    print("filtrado: ", df, df[hue].unique().tolist())
    bar_plot(df=df, base_dir=base_dir, file_name=filename, x_column=x_column, y_column=y_column, title=title, hue=hue, ax=ax, tipo='nt', hue_order=hue_order, palette=palette, x_order=x_order, sci=True, y_lim=True, y_max=100)

def groupb_by_plot(df):

    # df = df.sample(n=6, random_state=0, replace=True)
    accuracy = float(df['Accuracy (%)'].mean())

    return pd.DataFrame({'Accuracy (%)': [accuracy]})

def bar(df, base_dir):

    df = df[df['nt'].isin([0, 1, 2, 3, 7, 8, 9, 10])]
    print(df)
    # df = df[df['Round (t)'] >= 80]
    print(df['Round (t)'].value_counts())
    nt_list = df['nt'].tolist()
    for i in range(len(nt_list)):
        nt = nt_list[i]
        if int(nt) <= 3:
            nt = 'Updated'
        else:
            nt = 'Outdated'
        nt_list[i] = nt
    df['nt'] = np.array(nt_list)
    # df = df[df['Round (t)'] < 30]
    # df['nt'] = df['nt'].astype(int)
    print(df)
    df_test = df[
        ['Round (t)', 'Strategy', 'Accuracy (%)', 'Dataset',
         'nt']].groupby(['Round (t)', 'Strategy', 'Dataset', 'nt']).apply(
        lambda e: groupb_by_plot(e)).reset_index()
    print("agrupou plot")
    print(df_test)
    # figsize=(12, 9),
    sns.set(style='whitegrid')
    fig, axs = plt.subplots(3, 1, sharex='all', sharey='all', figsize=(6, 6))

    x_column = 'nt'
    y_column = 'Accuracy (%)'
    plt.xlabel(x_column)
    plt.ylabel(y_column)
    # ====================================================================
    dataset = 'EMNIST'
    title = """{}""".format(dataset)
    filename = 'nt'
    i = 0
    hue_order = ['FedAvg+FP$_{dc}$', 'FedAvg', 'FedYogi+FP$_{dc}$', "FedYogi", 'FedKD+FP$_{dc}$', "FedKD"]
    colors = ["mediumblue", "lightblue", "green", "lightgreen", "red", "mistyrose"]
    palette = {i: j for i, j in zip(hue_order, colors)}
    print(df_test['Strategy'].unique().tolist())
    hue = 'nt'
    x_order = ['Updated', 'Outdated']
    filter_and_plot(ax=axs[i], base_dir=base_dir, filename=filename, title=title, df=df_test,
                         dataset=dataset, x_column=x_column,
                         y_column=y_column,
                         hue='Strategy', x_order=x_order, hue_order=hue_order, style=None, palette=palette)
    axs[i].get_legend().remove()
    axs[i].set_xlabel('')
    axs[i].set_ylabel('')
    # ====================================================================
    dataset = 'CIFAR10'
    title = """CIFAR-10""".format()
    i = 1
    filter_and_plot(ax=axs[i], base_dir=base_dir, filename=filename, title=title, df=df_test,
                         dataset=dataset, x_column=x_column,
                         y_column=y_column,
                         hue='Strategy', x_order=x_order, hue_order=hue_order, style='nt', palette=palette)
    axs[i].get_legend().remove()
    axs[i].set_xlabel('')
    axs[i].set_ylabel('')
    # ====================================================================
    dataset = 'GTSRB'
    title = """GTSRB""".format()
    i = 2
    filter_and_plot(ax=axs[i], base_dir=base_dir, filename=filename, title=title, df=df_test,
                         dataset=dataset, x_column=x_column,
                         y_column=y_column,
                         hue='Strategy', x_order=x_order, hue_order=hue_order, style='nt', palette=palette)
    axs[i].get_legend().remove()
    axs[i].set_xlabel('')
    axs[i].set_ylabel('')
    # poc = pocs[2]
    # dataset = 'CIFAR10'
    # title = """CIFAR-10 ({})""".format(poc)
    # i = 1
    # j = 1
    # self.filter_and_plot(ax=axs[i, j], base_dir=base_dir, filename=filename, title=title, df=df_test,
    #                      experiment=experiment, dataset=dataset, poc=poc, x_column=x_column, y_column=y_column,
    #                      hue='Strategy')
    # axs[i, j].get_legend().remove()
    # axs[i, j].set_xlabel('')
    # axs[i, j].set_ylabel('')
    # ====================================================================
    # poc = pocs[2]
    # dataset = 'CIFAR10'
    # title = """CIFAR-10 ({})""".format(poc)
    # i = 1
    # j = 1
    # self.filter_and_plot(ax=axs[i, j], base_dir=base_dir, filename=filename, title=title, df=df_test,
    #                      experiment=experiment, dataset=dataset, poc=poc, x_column=x_column, y_column=y_column,
    #                      hue='Strategy')
    # legend = axs[i, j].get_legend()
    # print("legenda: ", legend)
    # # lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
    # # lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    # axs[i, j].get_legend().remove()
    # axs[i, j].set_xlabel('')
    # axs[i, j].set_ylabel('')
    # =========================///////////================================
    fig.suptitle("", fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.07, hspace=0.14)
    # plt.subplots_adjust(right=0.9)
    # fig.legend(
    #            loc="lower right")
    # fig.legend(lines, labels)
    # plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
    # fig.supxlabel(x_column, y=-0.02)
    fig.supylabel(y_column, x=-0.01)

    lines_labels = [axs[0].get_legend_handles_labels()]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    fig.legend(lines, labels, loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.06))
    fig.savefig(
        """{}alpha_dataset_round_{}_nt.png""".format(base_dir, y_column), bbox_inches='tight',
        dpi=400)
    fig.savefig(
        """{}alpha_dataset_round_{}_nt.svg""".format(base_dir, y_column), bbox_inches='tight',
        dpi=400)
    print("""{}alpha_dataset_round_{}_nt.png""".format(base_dir, y_column))


if __name__ == "__main__":
    # experiment_id = "1_new_clients"
    experiment_id = "1"
    cd = "false"
    total_clients = 20
    alphas = [0.1]
    dataset = ["EMNIST", "CIFAR10", "GTSRB"]
    # dataset = ["EMNIST", "CIFAR10"]
    # models_names = ["cnn_c"]
    model_name = "CNN_3"
    fraction_fit = 0.3
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
    solutions = ["FedAvg+FP", "FedYogi+FP", "FedAvg", "FedYogi", "FedKD+FP", "FedKD"]
    # solutions = ["MultiFedAvgWithFedPredict", "MultiFedAvg"]

    read_solutions = {solution: [] for solution in solutions}
    read_dataset_order = []
    for solution in solutions:
        for alpha in alphas:
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
                solution_file = solution
                if solution in ["FedAvg+FP", "FedYogi+FP"]:
                    solution_file = f"{solution}_dls_compredict"
                read_solutions[solution].append("""{}{}_{}_nt.csv""".format(read_path, dt, solution_file))

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
        fraction_fit,
        number_of_rounds,
        local_epochs)

    print(read_solutions)

    df, hue_order = read_data(read_solutions, read_dataset_order)
    print(df)

    bar(df,   write_path)