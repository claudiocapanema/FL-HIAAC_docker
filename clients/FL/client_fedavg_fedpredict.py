import sys
import logging
import os
import pickle

from utils.models_utils import load_model, get_weights, set_weights, test, train

logging.basicConfig(level=logging.INFO)  # Configure logging
logger = logging.getLogger(__name__)  # Create logger for the module

from fedpredict import fedpredict_client_torch
from clients.FL.client_fedavg import Client

# Make TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

class ClientFedAvgFP(Client):
    def __init__(self, args):
        try:
            super(ClientFedAvgFP, self).__init__(args)
            self.global_model = load_model(args.model[0], self.dataset, args.strategy, args.device)
            self.lt = 0
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
            results["Model size"] = self.models_size
            results["lt"] = self.lt
            return get_weights(self.model), len(self.trainloader.dataset), results
        except Exception as e:
            logger.error("fit error")
            logger.error("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))

    def evaluate(self, parameters, config):
        """Evaluate the utils on the data this client has."""
        try:
            logger.info("""eval cliente inicio fp""".format(config))
            t = config["t"]
            nt = t - self.lt
            # parameters = pickle.loads(config["parameters"])
            set_weights(self.global_model, parameters)
            combined_model = fedpredict_client_torch(local_model=self.model, global_model=self.global_model,
                                      t=t, T=100, nt=nt, device=self.device)
            loss, metrics = test(combined_model, self.valloader, self.device, self.client_id, t, self.dataset, self.n_classes)
            metrics["Model size"] = self.models_size
            metrics["Alpha"] = self.alpha
            logger.info("eval cliente fim fp")
            return loss, len(self.valloader.dataset), metrics
        except Exception as e:
            logger.error("evaluate error")
            logger.error("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))