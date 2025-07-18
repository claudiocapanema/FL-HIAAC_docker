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


if __name__ == "__main__":
    total_clients = 20
    alphas = [0.1, 1.0]
    dataset = ["EMNIST", "CIFAR10", "GTSRB"]

    df = read_data(total_clients, alphas, dataset)

    # table(df, write_path, "Accuracy (%)")