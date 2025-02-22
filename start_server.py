import argparse
import logging

import flwr as fl
import torch
from prometheus_client import start_http_server
from typing import List, Tuple
from flwr.common import Metrics

from flwr.common import (
    ndarrays_to_parameters,
)

from server.server_fedavg import FedAvg
from server.server_fedavg_fedpredict import FedAvgFP
from server.server_fedyogi import FedYogi
from server.server_fedyogi_fedpredict import FedYogiFP
from server.server_fedper import FedPer

# Initialize Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Parse command line arguments
parser = argparse.ArgumentParser(description="Flower Server")
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
    "--fraction_fit", type=float, default=1
)
parser.add_argument(
    "--batch_size", type=int, default=32
)
parser.add_argument(
    "--learning_rate", type=float, default=0.001
)

args = parser.parse_args()


from model.model import load_model, get_weights


# Define metric aggregation function
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["Accuracy"] for num_examples, m in metrics]
    balanced_accuracies = [num_examples * m["Balanced accuracy"] for num_examples, m in metrics]
    loss = [num_examples * m["Loss"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"Accuracy": sum(accuracies) / sum(examples), "Balanced accuracy": sum(balanced_accuracies) / sum(examples),
            "Loss": sum(loss) / sum(examples), "Round (t)": metrics[0][1]["Round (t)"], "Model size": metrics[0][1]["Model size"]}

def weighted_loss_avg(results: list[tuple[int, float]]) -> float:
    """Aggregate evaluation results obtained from multiple clients."""
    num_total_evaluation_examples = sum(num_examples for (num_examples, _) in results)
    weighted_losses = [num_examples * loss for num_examples, loss in results]
    return sum(weighted_losses) / num_total_evaluation_examples

# Function to Start Federated Learning Server
def start_fl_server(strategy, rounds):
    try:
        fl.server.start_server(
            server_address="0.0.0.0:8080",
            config=fl.server.ServerConfig(num_rounds=rounds),
            strategy=strategy,
        )
    except Exception as e:
        logger.error(f"FL Server error: {e}", exc_info=True)

def get_server(strategy_name):

    if strategy_name == "FedAvg":
        return FedAvg
    elif strategy_name == "FedAvg+FP":
        return FedAvgFP
    elif strategy_name == "FedYogi":
        return FedYogi
    elif strategy_name == "FedYogi+FP":
        return FedYogiFP
    elif strategy_name == "FedPer":
        return FedPer


# Main Function
if __name__ == "__main__":
    # Start Prometheus Metrics Server
    start_http_server(8000)

    # Initialize Strategy Instance and Start FL Serverstart_fl_server
    torch.random.manual_seed(0)
    ndarrays = get_weights(load_model(args.model, args.dataset, args.strategy))
    parameters = ndarrays_to_parameters(ndarrays)

    # Define the strategy
    strategy_ = get_server(args.strategy)
    strategy = strategy_(
        args=args,
        fraction_fit=args.fraction_fit,
        fraction_evaluate=1.0,
        min_available_clients=args.total_clients,
        evaluate_metrics_aggregation_fn=weighted_average,
        initial_parameters=parameters,
    )

    start_fl_server(strategy=strategy, rounds=args.number_of_rounds)
