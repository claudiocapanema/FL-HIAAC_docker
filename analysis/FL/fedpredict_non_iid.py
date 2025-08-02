from pathlib import Path
import numpy as np
import pandas as pd
import scipy.stats as st

import copy
import sys

from base_plots import bar_plot, line_plot, ecdf_plot
import matplotlib.pyplot as plt
from models_utils import load_data
import torch

def get_dataset_metrics(trainloader, client_id, n_classes):
    labels_me = []
    p_me = {i: 0 for i in range(n_classes)}
    with (torch.no_grad()):
        for batch in trainloader:
            labels = batch["label"]
            labels = labels.to("cuda:0")
            labels = labels.detach().cpu().numpy()
            labels_me += labels.tolist()
        unique, count = np.unique(labels_me, return_counts=True)
        data_unique_count_dict = dict(zip(np.array(unique).tolist(), np.array(count).tolist()))
        for label in data_unique_count_dict:
            p_me[label] = data_unique_count_dict[label]
        p_me = np.array(list(p_me.values()))
        fc_me = len(np.argwhere(p_me > 0)) / n_classes
        il_me = len(np.argwhere(p_me < np.sum(p_me) / n_classes)) / n_classes
        p_me = p_me / np.sum(p_me)
    return fc_me, il_me

def read_data(total_clients, alphas, datasets):

    batch_size = 32
    alphas_list = []
    datasets_list = []
    fc = []
    il = []
    for alpha in alphas:
        for dataset in datasets:
            for client_id in range(1, total_clients + 1):
                trainloader, valloader = load_data(
                    dataset_name=dataset,
                    alpha=alpha,
                    data_sampling_percentage=0.8,
                    partition_id=client_id,
                    num_partitions=total_clients + 1,
                    batch_size=batch_size,
                )
                n_classes = {'EMNIST': 47, 'MNIST': 10, 'CIFAR10': 10, 'GTSRB': 43, 'WISDM-W': 12, 'WISDM-P': 12, 'ImageNet': 15,
                     "ImageNet_v2": 15, "Gowalla": 7}[dataset]

                client_fc, client_il = get_dataset_metrics(trainloader, client_id, n_classes)
                alphas_list.append(alpha)
                datasets_list.append(dataset)
                fc.append(client_fc)
                il.append(client_il)

    df = pd.DataFrame({"Alpha": alphas_list, "Dataset": datasets_list, "FC": fc, "IL": il})
    print(df)
    df.to_csv(f"datasets_{datasets}.csv", index=False)

def line(df, base_dir, x, hue=None, style=None, ci=None, hue_order=None):

    datasets = df["Dataset"].unique().tolist()
    # datasets = ["ImageNet", "ImageNet"]
    alphas = df['\u03B1'].unique().tolist()

    fig, axs = plt.subplots(2, sharex='all', figsize=(9, 6))
    # hue_order = ["FedAvg", "FedYogi", "FedKD", "FedPer"]

    bar_plot(df=df, base_dir=base_dir, ax=axs[0],
              file_name="""solutions_{}""".format(datasets), x_column=x, y_column="Classes (%)",
              hue=hue, hue_order=hue_order, title="", tipo=None, y_lim=True, y_max=110)

            # if i == 0:
    # axs[0].get_legend().remove()

    bar_plot(df=df, base_dir=base_dir, ax=axs[1],
             file_name="""solutions_{}""".format(datasets), x_column=x, y_column="Imbalance level (%)",
             hue=hue, hue_order=hue_order, title="", tipo=None, y_lim=True, y_max=100)

    axs[1].get_legend().remove()

    # fig.suptitle("", fontsize=16)

    Path(base_dir).mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    # plt.subplots_adjust(wspace=0.2, hspace=0.3)
    fig.savefig(
        """{}non_iid_{}.png""".format(base_dir, datasets), bbox_inches='tight',
        dpi=400)
    fig.savefig(
        """{}non_iid_{}.svg""".format(base_dir, datasets), bbox_inches='tight',
        dpi=400)
    print("""{}non_iid_{}.svg""".format(base_dir, datasets))


if __name__ == "__main__":
    total_clients = 20
    alphas = [0.1, 0.5, 1.0]
    dataset = ["EMNIST", "CIFAR10", "GTSRB"]


    # df = read_data(total_clients, alphas, dataset)

    # table(df, write_path, "Accuracy (%)")

    df = pd.read_csv(f"datasets_{dataset}.csv")
    df["Dataset"] = np.array([d.replace("CIFAR10", "CIFAR-10") for d in df["Dataset"].tolist()])
    df["Classes (%)"] = df["FC"] * 100
    df["Imbalance level (%)"] = df["IL"] * 100
    df['\u03B1'] = df["Alpha"]
    # print(df)
    line(df, "", x='\u03B1', hue="Dataset", hue_order=["EMNIST", "CIFAR-10", "GTSRB"])
    line(df, "", x='\u03B1', hue="Dataset", hue_order=["EMNIST", "CIFAR-10", "GTSRB"])