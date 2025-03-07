import logging
import os

from utils.models_utils import load_model, get_weights, set_weights, test, train

logging.basicConfig(level=logging.INFO)  # Configure logging
logger = logging.getLogger(__name__)  # Create logger for the module

from fedpredict import fedpredict_client_torch
from clients.client_fedavg import Client

# Make TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

class ClientFedAvgFP(Client):
    def __init__(self, args):
        super(ClientFedAvgFP, self).__init__(args)
        self.global_model = load_model(args.model, args.dataset, args.strategy, args.device)
        self.lt = 0

    def fit(self, parameters, config):
        """Train the utils with data of this client."""

        logger.info("""fit cliente inicio fp config {}""".format(config))
        t = config['t']
        self.lt = t
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
        logger.info("fit cliente fim fp")
        return get_weights(self.model), len(self.trainloader.dataset), results

    def evaluate(self, parameters, config):
        """Evaluate the utils on the data this client has."""
        logger.info("""eval cliente inicio fp""".format(config))
        t = config["t"]
        nt = t - self.lt
        set_weights(self.global_model, parameters)
        combined_model = fedpredict_client_torch(local_model=self.model, global_model=self.global_model,
                                  t=t, T=100, nt=nt, device=self.device, fc=1, il=1)
        loss, metrics = test(combined_model, self.valloader, self.device, self.client_id, t, self.args.dataset, self.n_classes)
        metrics["Model size"] = self.models_size
        logger.info("eval cliente fim fp")
        return loss, len(self.valloader.dataset), metrics