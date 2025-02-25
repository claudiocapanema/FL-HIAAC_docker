import logging

import flwr as fl

from model.model import load_model, get_weights, load_data, set_weights, test, train
import torch

logging.basicConfig(level=logging.INFO)  # Configure logging
logger = logging.getLogger(__name__)  # Create logger for the module

class Client(fl.client.NumPyClient):
    def __init__(self, args):
        self.args = args
        self.model = load_model(args.model, args.dataset, args.strategy)
        logger.info("Preparing data...")
        logger.info("""args do cliente: {}""".format(self.args.client_id))
        self.client_id = args.client_id
        self.trainloader, self.valloader = load_data(
            dataset_name=self.args.dataset,
            alpha=self.args.alpha,
            data_sampling_percentage=self.args.data_percentage,
            partition_id=self.args.client_id,
            num_partitions=self.args.total_clients + 1,
            batch_size=self.args.batch_size,
        )
        logger.info("""leu dados {}""".format(self.args.client_id))

        self.local_epochs = self.args.local_epochs
        self.lr = self.args.learning_rate
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.lt = 0
        self.models_size = self._get_models_size()
        self.n_classes = {"EMNIST": 47, "CIFAR10": 10, "GTSRB": 43}[args.dataset]

    def fit(self, parameters, config):
        """Train the model with data of this client."""

        logger.info("""fit cliente inicio config {} device {}""".format(config, self.device))
        t = config['t']
        set_weights(self.model, parameters)
        results = train(
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
        return get_weights(self.model), len(self.trainloader.dataset), results

    def evaluate(self, parameters, config):
        """Evaluate the model on the data this client has."""
        logger.info("""eval cliente inicio""".format(config))
        t = config["t"]
        nt = t - self.lt
        set_weights(self.model, parameters)
        loss, metrics = test(self.model, self.valloader, self.device, self.client_id, t, self.args.dataset, self.n_classes)
        metrics["Model size"] = self.models_size
        logger.info("eval cliente fim")
        return loss, len(self.valloader.dataset), metrics

    def _get_models_size(self):
        parameters = [i.detach().cpu().numpy() for i in self.model.parameters()]
        size = 0
        for i in range(len(parameters)):
            size += parameters[i].nbytes
        return int(size)