import logging
import os
import sys

from clients.client_fedavg import Client

from utils.models_utils import get_weights_fedkd, set_weights_fedkd, test_fedkd, train_fedkd
import torch

logging.basicConfig(level=logging.INFO)  # Configure logging
logger = logging.getLogger(__name__)  # Create logger for the module

# Make TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

class ClientFedKD(Client):
    def __init__(self, args):
        super().__init__(args)
        self.lr_loss = torch.nn.MSELoss()
        self.round_of_last_fit = 0
        self.rounds_of_fit = 0
        self.accuracy_of_last_round_of_fit = 0
        self.start_server = 0
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9)
        feature_dim = 512
        self.W_h = torch.nn.Linear(feature_dim, feature_dim, bias=False)
        self.MSE = torch.nn.MSELoss()

    def fit(self, parameters, config):
        """Train the utils with data of this client."""
        try:
            logger.info("""fit cliente inicio config {} device {}""".format(config, self.device))
            t = config['t']
            set_weights_fedkd(self.model, parameters)
            results = train_fedkd(
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
            return get_weights_fedkd(self.model), len(self.trainloader.dataset), results

        except Exception as e:
            logger.info("fit")
            logger.info('Error on line {} {} {}'.format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))


    def evaluate(self, parameters, config):
        """Evaluate the utils on the data this client has."""
        logger.info("""eval cliente inicio""".format(config))
        t = config["t"]
        nt = t - self.lt
        # set_weights_fedkd(self.utils, parameters)
        loss, metrics = test_fedkd(self.model, self.valloader, self.device, self.client_id, t, self.args.dataset, self.n_classes)
        metrics["Model size"] = self.models_size
        logger.info("eval cliente fim")
        return loss, len(self.valloader.dataset), metrics