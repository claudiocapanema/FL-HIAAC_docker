import logging

import numpy as np
import tensorflow as tf
from flwr_datasets import FederatedDataset

logging.basicConfig(level=logging.INFO)  # Configure logging
logger = logging.getLogger(__name__)  # Create logger for the module
import torch.nn.functional as F
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor


# def load_data(data_sampling_percentage=0.5, client_id=1, total_clients=2):
#     """Load federated dataset partition based on client ID.
#
#     Args:
#         data_sampling_percentage (float): Percentage of the dataset to use for training.
#         client_id (int): Unique ID for the client.
#         total_clients (int): Total number of clients.
#
#     Returns:
#         Tuple of arrays: `(x_train, y_train), (x_test, y_test)`.
#     """
#
#     # Download and partition dataset
#     fds = FederatedDataset(dataset="cifar10", partitioners={"train": total_clients})
#     partition = fds.load_partition(client_id - 1, "train")
#     partition.set_format("numpy")
#
#     # Divide data on each client: 80% train, 20% test
#     partition = partition.train_test_split(test_size=0.2, seed=42)
#     x_train, y_train = partition["train"]["img"] / 255.0, partition["train"]["label"]
#     x_test, y_test = partition["test"]["img"] / 255.0, partition["test"]["label"]
#
#     # Apply data sampling
#     num_samples = int(data_sampling_percentage * len(x_train))
#     indices = np.random.choice(len(x_train), num_samples, replace=False)
#     x_train, y_train = x_train[indices], y_train[indices]
#
#     return (x_train, y_train), (x_test, y_test)

def load_data(partition_id: int, num_partitions: int, batch_size: int, data_sampling_percentage: int):
    """Load partition CIFAR10 data."""
    # Only initialize `FederatedDataset` once
    logger.info("Loading CIFAR10 data.", partition_id, num_partitions, batch_size, data_sampling_percentage)
    global fds
    if fds is None:
        partitioner = IidPartitioner(num_partitions=num_partitions)
        fds = FederatedDataset(
            dataset="uoft-cs/cifar10",
            partitioners={"train": partitioner},
        )
    partition = fds.load_partition(partition_id)
    # Divide data on each node: 80% train, 20% test
    test_size = 1 - data_sampling_percentage
    partition_train_test = partition.train_test_split(test_size=test_size, seed=42)
    pytorch_transforms = Compose(
        [ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    def apply_transforms(batch):
        """Apply transforms to the partition from FederatedDataset."""
        batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
        return batch

    partition_train_test = partition_train_test.with_transform(apply_transforms)
    trainloader = DataLoader(
        partition_train_test["train"], batch_size=batch_size, shuffle=True
    )
    testloader = DataLoader(partition_train_test["test"], batch_size=batch_size)
    return trainloader, testloader
