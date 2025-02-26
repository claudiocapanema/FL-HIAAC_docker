import logging

import flwr as fl

from utils.models_utils import load_model, get_weights, load_data, set_weights, test, train
import torch

logging.basicConfig(level=logging.INFO)  # Configure logging
logger = logging.getLogger(__name__)  # Create logger for the module

class ClientMEFL(fl.client.NumPyClient):
    def __init__(self, args):
        self.args = args
        self.model = [load_model(args.model[me], args.dataset[me], args.strategy) for me in args.model]
        self.ME = len(self.model)
        logger.info("Preparing data...")
        logger.info("""args do cliente: {}""".format(self.args.client_id))
        self.client_id = args.client_id
        self.trainloader = [None] * self.ME
        self.valloader = [None] * self.ME
        for me in range(self.ME):
            self.trainloader[me], self.valloader[me] = load_data(
                dataset_name=self.args.dataset[me],
                alpha=self.args.alpha[me],
                data_sampling_percentage=self.args.data_percentage,
                partition_id=self.args.client_id,
                num_partitions=self.args.total_clients + 1,
                batch_size=self.args.batch_size,
            )
        logger.info("""leu dados {}""".format(self.args.client_id))

        self.local_epochs = self.args.local_epochs
        self.lr = self.args.learning_rate
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.lt = [0] * self.ME
        self.models_size = self._get_models_size()
        self.n_classes = [{"EMNIST": 47, "CIFAR10": 10, "GTSRB": 43}[dataset] for dataset in self.args.dataset]

    def fit(self, parameters, config):
        """Train the model with data of this client."""

        logger.info("""fit cliente inicio config {} device {}""".format(config, self.device))
        t = config['t']
        me = config['me']
        self.lt[me] = t - self.lt[me]
        set_weights(self.model[me], parameters)
        results = train(
            self.model[me],
            self.trainloader[me],
            self.valloader[me],
            self.local_epochs,
            self.lr[me],
            self.device,
            self.client_id,
            t,
            self.args.dataset[me],
            self.n_classes[me]
        )
        results["me"] = me
        logger.info("fit cliente fim")
        return get_weights(self.model[me]), len(self.trainloader[me].dataset), results

    def evaluate(self, parameters, config):
        """Evaluate the model on the data this client has."""
        logger.info("""eval cliente inicio""".format(config))
        t = config["t"]
        me = config["me"]
        nt = t - self.lt[me]
        set_weights(self.model[me], parameters)
        loss, metrics = test(self.model[me], self.valloader[me], self.device, self.client_id, t, self.args.dataset[me], self.n_classes[me])
        metrics["Model size"] = self.models_size[me]
        metrics["me"] = me
        logger.info("eval cliente fim")
        return loss, len(self.valloader[me].dataset), metrics

    def _get_models_size(self):
        models_size = []
        for me in range(self.ME):
            parameters = [i.detach().cpu().numpy() for i in self.model[me].parameters()]
            size = 0
            for i in range(len(parameters)):
                size += parameters[i].nbytes
            models_size.append(int(size))

        return models_size