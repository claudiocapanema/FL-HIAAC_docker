import argparse
import logging
import os

import flwr as fl
import tensorflow as tf

from model.model import Net, get_weights, load_data, set_weights, test, train
import torch
import copy

logging.basicConfig(level=logging.INFO)  # Configure logging
logger = logging.getLogger(__name__)  # Create logger for the module

from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor
from fedpredict import fedpredict_client_torch

# Make TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Parse command line arguments
parser = argparse.ArgumentParser(description="Flower client")

parser.add_argument(
    "--total_clients", type=int, default=2, help="Total clients to spawn (default: 2)"
)
parser.add_argument(
    "--number_of_rounds", type=int, default=5, help="Number of FL rounds (default: 100)"
)
parser.add_argument(
    "--data_percentage",
    type=float,
    default=0.6,
    help="Portion of client data to use (default: 0.6)",
)
parser.add_argument(
    "--strategy", type=str, default='FedAvg+FP', help="Strategy to use (default: FedAvg)"
)
parser.add_argument(
    "--alpha", type=float, default=0.1, help="Dirichlet alpha"
)
parser.add_argument(
    "--round_new_clients", type=float, default=0.1, help=""
)
parser.add_argument(
    "--fraction_new_clients", type=float, default=0.1, help=""
)
parser.add_argument(
    "--local_epochs", type=float, default=1, help=""
)
parser.add_argument(
    "--dataset", type=str, default="CIFAR10"
)
parser.add_argument(
    "--model", type=str, default=""
)
parser.add_argument(
    "--cd", type=str, default="false"
)
parser.add_argument(
    "--server_address", type=str, default="server:8080"
)
parser.add_argument(
    "--fraction_fit", type=float, default=1
)
parser.add_argument(
    "--client_id", type=int, default=1
)
parser.add_argument(
    "--batch_size", type=int, default=32
)
parser.add_argument(
    "--learning_rate", type=float, default=0.001
)


args = parser.parse_args()

# Create an instance of the model and pass the learning rate as an argument

fds = None

class Client(fl.client.NumPyClient):
    def __init__(self, args):
        self.args = args
        self.model = Net()
        self.global_model = Net()
        logger.info("Preparing data...")
        logger.info("""args do cliente: {}""".format(self.args.client_id))
        self.client_id = args.client_id
        self.trainloader, self.valloader = self.load_data(
            data_sampling_percentage=self.args.data_percentage,
            partition_id=self.args.client_id,
            num_partitions=self.args.total_clients+1,
            batch_size=self.args.batch_size,
        )
        logger.info("""leu dados {}""".format(self.args.client_id))

        self.contar = 0

        self.local_epochs = 1
        self.lr = self.args.learning_rate
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.lt = 0
        self.models_size = self._get_models_size()

    def fit(self, parameters, config):
        """Train the model with data of this client."""

        logger.info("""fit cliente inicio fp config {}""".format(config))
        t = config['t']
        self.lt = t - self.lt
        set_weights(self.model, parameters)
        results = train(
            self.model,
            self.trainloader,
            self.valloader,
            self.local_epochs,
            self.lr,
            self.device,
            self.client_id,
            t
        )
        logger.info("fit cliente fim fp")
        return get_weights(self.model), len(self.trainloader.dataset), results

    def evaluate(self, parameters, config):
        """Evaluate the model on the data this client has."""
        logger.info("""eval cliente inicio fp""".format(config))
        t = config["t"]
        nt = t - self.lt
        set_weights(self.global_model, parameters)
        combined_model = fedpredict_client_torch(local_model=self.model, global_model=self.global_model,
                                  t=t, T=100, nt=nt, device=self.device, fc=1, il=1)
        loss, metrics = test(combined_model, self.valloader, self.device, self.client_id, t)
        metrics["Model size"] = self.models_size
        logger.info("eval cliente fim fp")
        return loss, len(self.valloader.dataset), metrics

    def load_data(self, partition_id: int, num_partitions: int, batch_size: int, data_sampling_percentage: int):
        """Load partition CIFAR10 data."""
        # Only initialize `FederatedDataset` once
        logger.info("""Loading CIFAR10 data.""".format(partition_id, num_partitions, batch_size, data_sampling_percentage))
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

    def _get_models_size(self):
        parameters = [i.detach().cpu().numpy() for i in self.model.parameters()]
        size = 0
        for i in range(len(parameters)):
            size += parameters[i].nbytes
        return int(size)




# Function to Start the Client
def start_fl_client():
    try:
        client = Client(args).to_client()
        fl.client.start_client(server_address=args.server_address, client=client)
    except Exception as e:
        logger.error("Error starting FL client: %s", e)
        return {"status": "error", "message": str(e)}


if __name__ == "__main__":
    # Call the function to start the client
    start_fl_client()
