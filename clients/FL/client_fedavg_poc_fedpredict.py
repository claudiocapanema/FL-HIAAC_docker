import sys
import logging

import flwr as fl

from utils.models_utils import load_model, get_weights, load_data, set_weights, test, train
import torch
from clients.FL.client_fedavg_fedpredict import ClientFedAvgFP
from fedpredict import fedpredict_client_torch

logging.basicConfig(level=logging.INFO)  # Configure logging
logger = logging.getLogger(__name__)  # Create logger for the module

class ClientPOCFP(ClientFedAvgFP):
    def __init__(self, args):
        try:
            super(ClientPOCFP, self).__init__(args)

        except Exception as e:
            logger.error("__init__ error")
            logger.error("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))

    def fit(self, parameters, config):
        """Train the utils with data of this client."""
        try:
            logger.info("""fit cliente inicio fp config {}""".format(config))
            t = config['t']
            self.lt = t
            if len(parameters) > 0:
                set_weights(self.model, parameters)
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
            logger.info("fit cliente fim fp")
            self.models_size = self._get_models_size()
            results["Model size"] = self.models_size
            logger.info(f"model size: {self.models_size}")
            results["lt"] = self.lt
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
            combined_model = fedpredict_client_torch(local_model=self.model, global_model=parameters,
                                                     t=t, T=self.T, nt=nt, device=self.device,
                                                     global_model_original_shape=self.model_shape)
            loss, metrics = test(combined_model, self.valloader, self.device, self.client_id, t, self.dataset,
                                 self.n_classes)
            self.models_size = self._get_models_size(parameters)
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