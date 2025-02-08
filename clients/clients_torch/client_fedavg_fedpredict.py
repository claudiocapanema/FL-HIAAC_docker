import argparse
import logging
import os

import flwr as fl
import tensorflow as tf

from ..model.model import Net, get_weights, load_data, set_weights, test, train
import torch

logging.basicConfig(level=logging.INFO)  # Configure logging
logger = logging.getLogger(__name__)  # Create logger for the module
from clients.clients_torch.client import Client

# Make TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Parse command line arguments
parser = argparse.ArgumentParser(description="Flower client")

parser.add_argument(
    "--server_address", type=str, default="server:8080", help="Address of the server"
)
parser.add_argument(
    "--batch_size", type=int, default=32, help="Batch size for training"
)
parser.add_argument(
    "--learning_rate", type=float, default=0.1, help="Learning rate for the optimizer"
)
parser.add_argument("--client_id", type=int, default=1, help="Unique ID for the client")
parser.add_argument(
    "--total_clients", type=int, default=3, help="Total number of clients"
)
parser.add_argument(
    "--data_percentage", type=float, default=0.5, help="Portion of client data to use"
)

args = parser.parse_args()

# Create an instance of the model and pass the learning rate as an argument


class ClientFedPredict(Client):
    def __init__(self, args):
        self.args = args
        self.net = Net()
        logger.info("Preparing data fedpredict...")
        logger.info("""args do cliente: {}""".format(self.args.client_id))
        # (x_train, y_train), (x_test, y_test) = load_data(
        #     data_sampling_percentage=self.args.data_percentage,
        #     client_id=self.args.client_id,
        #     total_clients=self.args.total_clients,
        # )
        self.trainloader, self.valloader = self.load_data(
            data_sampling_percentage=self.args.data_percentage,
            partition_id=self.args.client_id,
            num_partitions=self.args.total_clients+1,
            batch_size=self.args.batch_size,
        )
        logger.info("leu dados")

        self.contar = 0

        self.local_epochs = 1
        self.lr = self.args.learning_rate
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def fit(self, parameters, config):
        """Train the model with data of this client."""
        logger.info("fit cliente inicio fedpredict")
        set_weights(self.net, parameters)
        results = train(
            self.net,
            self.trainloader,
            self.valloader,
            self.local_epochs,
            self.lr,
            self.device,
        )
        logger.info("fit cliente fim fedpredict")
        return get_weights(self.net), len(self.trainloader.dataset), results

    def evaluate(self, parameters, config):
        """Evaluate the model on the data this client has."""
        logger.info("eval cliente inicio fedpredict")
        set_weights(self.net, parameters)
        loss, accuracy = test(self.net, self.valloader, self.device)
        logger.info("eval cliente fim fedpredict")
        return loss, len(self.valloader.dataset), {"accuracy": accuracy}


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
