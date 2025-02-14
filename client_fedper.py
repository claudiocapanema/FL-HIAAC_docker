import argparse
import logging
import os

import flwr as fl
import tensorflow as tf
from collections import OrderedDict

from model.model import test, train
import torch
import copy

logging.basicConfig(level=logging.INFO)  # Configure logging
logger = logging.getLogger(__name__)  # Create logger for the module

from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner, DirichletPartitioner
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor
from client_fedavg import Client

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

def get_weights(net):
    return [val.cpu().numpy() for _, val in net.state_dict().items()][:-2]


def set_weights(net, parameters):
    head = [val.cpu().numpy() for _, val in net.state_dict().items()][-2:]
    parameters += head
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)

class ClientFedPer(Client):
    def __init__(self, args):
        super().__init__(args)

    def _get_models_size(self):
        parameters = [i.detach().cpu().numpy() for i in self.model.parameters()]
        size = 0
        for i in range(len(parameters)-2):
            size += parameters[i].nbytes
        return int(size)




# Function to Start the Client
def start_fl_client():
    try:
        client = ClientFedPer(args).to_client()
        fl.client.start_client(server_address=args.server_address, client=client)
    except Exception as e:
        logger.error("Error starting FL client: %s", e)
        return {"status": "error", "message": str(e)}


if __name__ == "__main__":
    # Call the function to start the client
    start_fl_client()
