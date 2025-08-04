import sys
import argparse
import logging

import flwr as fl
import numpy as np
import torch
from prometheus_client import start_http_server
from typing import List, Tuple
from flwr.common import Metrics

from utils.models_utils import load_model, get_weights, get_weights_fedkd, load_data, set_weights, test, train

from server.FL.server_fedavg import FedAvg
from server.FL.server_fedavg_fedpredict import FedAvgFP
from server.FL.server_fedavg_poc import FedAvgPOC
from server.FL.server_fedavg_poc_fedpredict import FedAvgPOCFP
from server.FL.server_fedavg_rawcs import FedAvgRAWCS
from server.FL.server_fedavg_rawcs_fedpredict import FedAvgRAWCSFP
from server.FL.server_fedyogi import FedYogi
from server.FL.server_fedyogi_fedpredict import FedYogiFP
from server.FL.server_fedkd import FedKD
from server.FL.server_fedkd_fedpredict import FedKDFP
from server.FL.server_fedper import FedPer
from server.MEFL.server_multifedavg import MultiFedAvg
from server.MEFL.server_multifedefficiency import MultiFedEfficiency
from server.MEFL.server_multifedavgrr import MultiFedAvgRR
from server.MEFL.server_fedfairmmfl import FedFairMMFL
from server.MEFL.server_multifedavg_fedpredict_dynamic import MultiFedAvgFedPredictDynamic
from server.MEFL.server_multifedavg_multifedpredict import MultiFedAvgMultiFedPredict
from server.MEFL.server_multifedavg_fedpredict import MultiFedAvgFedPredict

from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)

# Initialize Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description="Generated Docker Compose")
parser.add_argument(
    "--total_clients", type=int, default=20, help="Total clients to spawn (default: 2)"
)
parser.add_argument(
    "--number_of_rounds", type=int, default=5, help="Number of FL rounds (default: 5)"
)
parser.add_argument(
    "--data_percentage",
    type=float,
    default=0.8,
    help="Portion of client data to use (default: 0.6)",
)
parser.add_argument(
    "--random", action="store_true", help="Randomize client configurations"
)

parser.add_argument(
    "--strategy", type=str, default='FedAvg', help="Strategy to use (default: FedAvg)"
)
parser.add_argument(
    "--alpha", action="append", help="Dirichlet alpha"
)
parser.add_argument(
    "--experiment_id", type=str, default="", help=""
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
    "--dataset", action="append"
)
parser.add_argument(
    "--model", action="append"
)
parser.add_argument(
    "--cd", type=str, default="false"
)
parser.add_argument(
    "--fraction_fit", type=float, default=0.3
)
parser.add_argument(
    "--client_id", type=int, default=1
)
parser.add_argument(
    "--batch_size", type=int, default=32
)
parser.add_argument(
    "--learning_rate", type=float, default=0.01
)
parser.add_argument(
    "--tw", type=int, default=15, help="TW window of rounds used in MultiFedEfficiency"
)
parser.add_argument(
    "--reduction", type=int, default=3, help="Reduction in the number of training clients used in MultiFedEfficiency"
)
parser.add_argument(
    "--df", type=float, default=0, help="Free budget redistribution factor used in MultiFedEfficiency"
)
parser.add_argument(
    "--compression", type=str, default=""
)

parser.add_argument(
    "--device", type=str, default="cuda"
)

args = parser.parse_args()


# Define metric aggregation function
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    try:
        # Multiply accuracy of each client by number of examples used
        accuracies = [num_examples * m["Accuracy"] for num_examples, m in metrics]
        balanced_accuracies = [num_examples * m["Balanced accuracy"] for num_examples, m in metrics]
        loss = [num_examples * m["Loss"] for num_examples, m in metrics]
        examples = [num_examples for num_examples, _ in metrics]
        model_size = int(np.mean([m["Model size"] for num_examples, m in metrics]))

        # Aggregate and return custom metric (weighted average)
        return {"Accuracy": sum(accuracies) / sum(examples), "Balanced accuracy": sum(balanced_accuracies) / sum(examples),
                "Loss": sum(loss) / sum(examples), "Round (t)": metrics[0][1]["Round (t)"], "Model size": model_size, "Alpha": metrics[0][1]["Alpha"]}
    except Exception as e:
        logger.error("weighted_average error")
        logger.error("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))


def weighted_average_fit(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    try:
        # Multiply accuracy of each client by number of examples used
        logger.info(f"metricas recebidas: {metrics}")
        accuracies = [num_examples * m["train_accuracy"] for num_examples, m in metrics]
        balanced_accuracies = [num_examples * m["train_balanced_accuracy"] for num_examples, m in metrics]
        loss = [num_examples * m["train_loss"] for num_examples, m in metrics]
        examples = [num_examples for num_examples, _ in metrics]

        # Aggregate and return custom metric (weighted average)
        return {"Accuracy": sum(accuracies) / sum(examples), "Balanced accuracy": sum(balanced_accuracies) / sum(examples),
                "Loss": sum(loss) / sum(examples), "Round (t)": metrics[0][1]["Round (t)"], "Model size": metrics[0][1]["Model size"]}
    except Exception as e:
        logger.error("weighted_average_fit error")
        logger.error("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))

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
        logger.error("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))

def get_server(strategy_name):

    if strategy_name == "FedAvg":
        return FedAvg
    elif strategy_name == "FedAvg+FP":
        return FedAvgFP
    elif strategy_name == "FedAvgPOC":
        return FedAvgPOC
    elif strategy_name == "FedAvgPOC+FP":
        return FedAvgPOCFP
    elif strategy_name == "FedAvgRAWCS":
        return FedAvgRAWCS
    elif strategy_name == "FedAvgRAWCS+FP":
        return FedAvgRAWCSFP
    elif strategy_name == "FedYogi":
        return FedYogi
    elif strategy_name == "FedYogi+FP":
        return FedYogiFP
    elif strategy_name == "FedPer":
        return FedPer
    elif strategy_name == "FedKD":
        return FedKD
    elif strategy_name == "FedKD+FP":
        return FedKDFP
    elif strategy_name == "MultiFedAvg":
        return MultiFedAvg
    elif strategy_name == "MultiFedEfficiency":
        return MultiFedEfficiency
    elif strategy_name == "MultiFedAvgRR":
        return MultiFedAvgRR
    elif strategy_name == "FedFairMMFL":
        return FedFairMMFL
    elif strategy_name == "MultiFedAvg+FPD":
        return MultiFedAvgFedPredictDynamic
    elif strategy_name == "MultiFedAvg+MFP":
        return MultiFedAvgMultiFedPredict
    elif strategy_name == "MultiFedAvg+FP":
        return MultiFedAvgFedPredict
    else:
        raise Exception(f"Invalid strategy name: {strategy_name}")

# Main Function
if __name__ == "__main__":
    # Start Prometheus Metrics Server
    start_http_server(8000)

    # Initialize Strategy Instance and Start FL Serverstart_fl_server
    if "FedKD" in args.strategy:
        initial_parameters = ndarrays_to_parameters(get_weights_fedkd(load_model(args.model[0], args.dataset[0], args.strategy, args.device)))
    else:
        initial_parameters = ndarrays_to_parameters(get_weights(load_model(args.model[0], args.dataset[0], args.strategy, args.device)))


    torch.random.manual_seed(0)
    logger.info(f"argumentos recebidos: {args}")
    # Define the strategy
    strategy_ = get_server(args.strategy)
    strategy = strategy_(
        args=args,
        fraction_fit=args.fraction_fit,
        fraction_evaluate=1.0,
        min_available_clients=args.total_clients,
        evaluate_metrics_aggregation_fn=weighted_average,
        fit_metrics_aggregation_fn=weighted_average_fit,
        # initial_parameters=None,
        initial_parameters=initial_parameters,
    )

    start_fl_server(strategy=strategy, rounds=args.number_of_rounds)
