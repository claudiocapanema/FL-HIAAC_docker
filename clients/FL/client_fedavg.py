import sys
import logging

import flwr as fl

from utils.models_utils import load_model, get_weights, load_data, set_weights, test, train
import torch

logging.basicConfig(level=logging.INFO)  # Configure logging
logger = logging.getLogger(__name__)  # Create logger for the module

class Client(fl.client.NumPyClient):
    def __init__(self, args):
        try:
            self.args = args
            self.dataset = self.args.dataset[0]
            self.model = load_model(args.model[0], self.dataset, args.strategy, args.device)
            self.alpha = float(self.args.alpha[0])
            logger.info("Preparing data...")
            logger.info("""args do cliente: {} {}""".format(self.args.client_id, self.alpha))
            self.client_id = args.client_id
            self.trainloader, self.valloader = load_data(
                dataset_name=self.dataset,
                alpha=self.alpha,
                data_sampling_percentage=self.args.data_percentage,
                partition_id=self.args.client_id,
                num_partitions=self.args.total_clients + 1,
                batch_size=self.args.batch_size,
            )
            self.optimizer = self._get_optimizer(dataset_name=self.dataset)
            logger.info("""leu dados client id {}""".format(self.args.client_id))

            self.local_epochs = self.args.local_epochs
            self.lr = self.args.learning_rate
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            self.lt = 0
            self.models_size = self._get_models_size()
            self.n_classes = \
            {'EMNIST': 47, 'MNIST': 10, 'CIFAR10': 10, 'GTSRB': 43, 'WISDM-W': 12, 'WISDM-P': 12, 'ImageNet': 15,
             "ImageNet_v2": 15, "Gowalla": 7}[self.args.dataset[0]]

        except Exception as e:
            logger.error("__init__ error")
            logger.error("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))

    def fit(self, parameters, config):
        try:
            """Train the utils with data of this client."""

            logger.info("""fit cliente inicio config {} device {}""".format(config, self.device))
            t = config['t']
            if len(parameters) > 0:
                set_weights(self.model, parameters)
            self.optimizer = self._get_optimizer(dataset_name=self.dataset)
            results = train(
                self.model,
                self.trainloader,
                self.valloader,
                self.optimizer,
                self.local_epochs,
                self.lr,
                self.device,
                self.client_id,
                t,
                self.dataset,
                self.n_classes
            )
            logger.info("fit cliente fim")
            return get_weights(self.model), len(self.trainloader.dataset), results
        except Exception as e:
            logger.error("fit error")
            logger.error("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))

    def evaluate(self, parameters, config):
        try:
            """Evaluate the utils on the data this client has."""
            logger.info("""eval cliente inicio""".format(config))
            t = config["t"]
            nt = t - self.lt
            set_weights(self.model, parameters)
            loss, metrics = test(self.model, self.valloader, self.device, self.client_id, t, self.dataset, self.n_classes)
            metrics["Model size"] = self.models_size
            logger.info("eval cliente fim")
            return loss, len(self.valloader.dataset), metrics
        except Exception as e:
            logger.error("evaluate error")
            logger.error("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))

    def _get_models_size(self):
        try:
            parameters = [i.detach().cpu().numpy() for i in self.model.parameters()]
            size = 0
            for i in range(len(parameters)):
                size += parameters[i].nbytes
            return int(size)
        except Exception as e:
            logger.error("_get_models_size error")
            logger.error("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))

    def _get_optimizer(self, dataset_name):
        try:
            return {
                    'EMNIST': torch.optim.SGD(self.model.parameters(), lr=self.args.learning_rate, momentum=0.9),
                    'MNIST': torch.optim.SGD(self.model.parameters(), lr=self.args.learning_rate, momentum=0.9),
                    'CIFAR10': torch.optim.SGD(self.model.parameters(), lr=self.args.learning_rate, momentum=0.9),
                    'GTSRB': torch.optim.SGD(self.model.parameters(), lr=self.args.learning_rate, momentum=0.9),
                    'WISDM-W':torch.optim.RMSprop(self.model.parameters(), lr=0.001),
                    'WISDM-P': torch.optim.RMSprop(self.model.parameters(), lr=0.001),
                    'ImageNet100': torch.optim.SGD(self.model.parameters(), lr=self.args.learning_rate, momentum=0.9),
                    'ImageNet': torch.optim.Adam(self.model.parameters(), lr=0.01),
                    "ImageNet_v2": torch.optim.Adam(self.model.parameters(), lr=0.01),
                    "Gowalla": torch.optim.RMSprop(self.model.parameters(), lr=0.001)}[dataset_name]
        except Exception as e:
            logger.error("_get_optimizer error")
            logger.error("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))