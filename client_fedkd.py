import argparse
import logging
import sys
import os
import numpy as np
import torch.nn.functional as F

import flwr as fl
import tensorflow as tf
from client_fedavg import Client
from sklearn import metrics
from sklearn.preprocessing import label_binarize

from model.model import load_model, get_weights_fedkd, load_data, set_weights_fedkd, test_fedkd, train_fedkd
import torch
import copy

logging.basicConfig(level=logging.INFO)  # Configure logging
logger = logging.getLogger(__name__)  # Create logger for the module

from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner, DirichletPartitioner
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
    default=0.8,
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

class ClientFedKD(Client):
    def __init__(self, args):
        super().__init__(args)
        self.lr_loss = torch.nn.MSELoss()
        self.round_of_last_fit = 0
        self.rounds_of_fit = 0
        self.T = int(args.T)
        self.accuracy_of_last_round_of_fit = 0
        self.start_server = 0
        self.n_rate = float(args.n_rate)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=0.9)
        self.fedkd_model_filename = """./{}_saved_weights/{}/{}/model.pth""".format(self.strategy_name.lower(),
                                                                                    self.model_name, self.cid)
        feature_dim = 512
        self.W_h = torch.nn.Linear(feature_dim, feature_dim, bias=False)
        self.MSE = torch.nn.MSELoss()

    def fit(self, parameters, config):
        """Train the model with data of this client."""

        logger.info("""fit cliente inicio config {} device {}""".format(config, self.device))
        t = config['t']
        self.lt = t - self.lt
        set_weights_fedkd(self.model, parameters)
        results = train_fedkd(
            self.model,
            self.trainloader,
            self.valloader,
            self.local_epochs,
            self.lr,
            self.device,
            self.client_id,
            t,
            self.args.dataset,
            self.n_classes
        )
        logger.info("fit cliente fim")
        return get_weights_fedkd(self.model), len(self.trainloader.dataset), results

    def evaluate(self, parameters, config):
        """Evaluate the model on the data this client has."""
        logger.info("""eval cliente inicio""".format(config))
        t = config["t"]
        nt = t - self.lt
        set_weights_fedkd(self.model, parameters)
        loss, metrics = test_fedkd(self.model, self.valloader, self.device, self.client_id, t, self.args.dataset, self.n_classes)
        metrics["Model size"] = self.models_size
        logger.info("eval cliente fim")
        return loss, len(self.valloader.dataset), metrics


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
