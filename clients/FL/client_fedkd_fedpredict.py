import logging
import os
import sys

from clients.FL.client_fedkd import ClientFedKD

from utils.models_utils import test_fedkd_fedpredict

logging.basicConfig(level=logging.INFO)  # Configure logging
logger = logging.getLogger(__name__)  # Create logger for the module

# Make TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

class ClientFedKDFedPredict(ClientFedKD):
    def __init__(self, args):
        super().__init__(args)
        self.lt = 0

    def fit(self, parameters, config):
        """Train the utils with data of this client."""
        try:
            t = config['t']
            self.lt = t
            return super().fit(parameters, config)

        except Exception as e:
            logger.info("fit")
            logger.info('Error on line {} {} {}'.format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))


    def evaluate(self, parameters, config):
        """Evaluate the utils on the data this client has."""
        logger.info("""eval cliente inicio""".format(config))
        t = config["t"]
        nt = t - self.lt
        # set_weights_fedkd(self.utils, parameters)
        loss, metrics = test_fedkd_fedpredict(self.lt, self.model, self.valloader, self.device, self.client_id, t, self.dataset, self.n_classes)
        metrics["Model size"] = self.models_size
        logger.info("eval cliente fim")
        return loss, len(self.valloader.dataset), metrics