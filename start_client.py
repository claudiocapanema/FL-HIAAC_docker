import argparse
import logging
import os

import flwr as fl
from clients.client_fedavg import Client
from clients.client_fedavg_fedpredict import ClientFedAvgFP
from clients.client_fedper import ClientFedPer
from clients.client_fedkd import ClientFedKD
from clients.client_fedyogi import ClientFedYogi
from clients.client_fedyogi_fedpredict import ClientFedYogiFP

logging.basicConfig(level=logging.INFO)  # Configure logging
logger = logging.getLogger(__name__)  # Create logger for the module


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

def get_client(strategy_name):

    if strategy_name in "FedAvg":
        return Client
    elif strategy_name == "FedAvg+FP":
        return ClientFedAvgFP
    elif strategy_name == "FedYogi":
        return ClientFedYogi
    elif strategy_name == "FedYogi+FP":
        return ClientFedYogiFP
    elif strategy_name == "FedPer":
        return ClientFedPer
    elif strategy_name == "FedKD":
        return ClientFedKD


# Function to Start the Client
def start_fl_client():
    try:
        client_ = get_client(args.strategy)
        client = client_(args).to_client()
        fl.client.start_client(server_address=args.server_address, client=client)
    except Exception as e:
        logger.error("Error starting FL client: %s", e)
        return {"status": "error", "message": str(e)}


if __name__ == "__main__":
    # Call the function to start the client
    start_fl_client()
