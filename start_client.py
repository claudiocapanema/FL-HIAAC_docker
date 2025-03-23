import sys
import argparse
import logging

import flwr as fl
from clients.FL.client_fedavg import Client
from clients.FL.client_fedavg_fedpredict import ClientFedAvgFP
from clients.FL.client_fedper import ClientFedPer
from clients.FL.client_fedkd import ClientFedKD
from clients.FL.client_fedkd_fedpredict import ClientFedKDFedPredict
from clients.FL.client_fedyogi import ClientFedYogi
from clients.FL.client_fedyogi_fedpredict import ClientFedYogiFP
from clients.MEFL.client_multifedavg import ClientMultiFedAvg
from clients.MEFL.client_multifedefficency import ClientMultiFedEfficiency
from clients.MEFL.client_multifedavgrr import ClientMultiFedAvgRR
from clients.MEFL.client_fedfairmmfl import ClientFedFairMMFL
from clients.MEFL.client_multifedavg_multifedpredict import ClientMultiFedAvgMultiFedPredict
from clients.MEFL.client_multifedavg_fedpredict_dynamic import ClientMultiFedAvgFedPredictDynamic

logging.basicConfig(level=logging.INFO)  # Configure logging
logger = logging.getLogger(__name__)  # Create logger for the module


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
    "--concept_drift_experiment_id", type=int, default=0, help=""
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
    "--server_address", type=str, default="server:8080"
)

parser.add_argument(
    "--device", type=str, default="cuda"
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
    elif strategy_name == "FedKD+FP":
        return ClientFedKDFedPredict
    elif strategy_name == "MultiFedAvg":
        return ClientMultiFedAvg
    elif strategy_name == "MultiFedEfficiency":
        return ClientMultiFedEfficiency
    elif strategy_name == "MultiFedAvgRR":
        return ClientMultiFedAvgRR
    elif strategy_name == "FedFairMMFL":
        return ClientFedFairMMFL
    elif strategy_name == "MultiFedAvg+FPD":
        return ClientMultiFedAvgFedPredictDynamic
    elif strategy_name == "MultiFedAvg+MFP":
        return ClientMultiFedAvgMultiFedPredict

# Function to Start the Client
def start_fl_client():
    try:
        client_ = get_client(args.strategy)
        client = client_(args).to_client()
        fl.client.start_client(server_address=args.server_address, client=client)
    except Exception as e:
        logger.error("Error starting FL client: %s", e)
        logger.error("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))
        return {"status": "error", "message": str(e)}


if __name__ == "__main__":
    # Call the function to start the client
    start_fl_client()
