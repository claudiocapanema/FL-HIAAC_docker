import sys
import logging

import flwr as fl

from utils.models_utils import load_model, get_weights, load_data, set_weights, test, train
import torch
from clients.FL.client_fedavg import Client

logging.basicConfig(level=logging.INFO)  # Configure logging
logger = logging.getLogger(__name__)  # Create logger for the module

class ClientPOC(Client):
    def __init__(self, args):
        try:
            super(ClientPOC, self).__init__(args)

        except Exception as e:
            logger.error("__init__ error")
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
            metrics["Alpha"] = self.alpha
            loss_train, metrics_train = test(self.model, self.trainloader, self.device, self.client_id, t, self.dataset,
                                           self.n_classes)
            metrics["train_loss"] = loss_train
            metrics["train_samples"] = len(self.trainloader.dataset)
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