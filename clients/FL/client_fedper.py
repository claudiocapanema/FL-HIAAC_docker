import sys
import logging
import os
from torch.nn.parameter import Parameter

from utils.models_utils import test, train
import torch

logging.basicConfig(level=logging.INFO)  # Configure logging
logger = logging.getLogger(__name__)  # Create logger for the module

from clients.FL.client_fedavg import Client

# Make TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

def get_weights(net):
    return [val.cpu().numpy() for _, val in net.state_dict().items()][:-2]


def set_weights(net, parameters):
    head = [val.cpu().numpy() for _, val in net.state_dict().items()][-2:]
    parameters += head
    # params_dict = zip(net.state_dict().keys(), parameters)
    # state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    # net.load_state_dict(state_dict, strict=True)
    parameters = [Parameter(torch.Tensor(i.tolist())) for i in parameters]
    for new_param, old_param in zip(parameters, net.parameters()):
        old_param.data = new_param.data.clone()

class ClientFedPer(Client):
    def __init__(self, args):
        try:
            super().__init__(args)
        except Exception as e:
            logger.error("__init__ error")
            logger.error("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))

    def fit(self, parameters, config):
        """Train the utils with data of this client."""
        try:
            logger.info("""fit cliente inicio config {} device {}""".format(config, self.device))
            t = config['t']
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
            results["Model size"] = self.models_size
            logger.info("fit cliente fim")
            return get_weights(self.model), len(self.trainloader.dataset), results
        except Exception as e:
            logger.error("fit error")
            logger.error("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))

    def evaluate(self, parameters, config):
        """Evaluate the utils on the data this client has."""
        try:
            logger.info("""eval cliente inicio""".format(config))
            t = config["t"]
            nt = t - self.lt
            set_weights(self.model, parameters)
            loss, metrics = test(self.model, self.valloader, self.device, self.client_id, t, self.dataset, self.n_classes)
            metrics["Model size"] = self.models_size
            metrics["Alpha"] = self.alpha
            logger.info("eval cliente fim")
            return loss, len(self.valloader.dataset), metrics
        except Exception as e:
            logger.error("evaluate error")
            logger.error("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))

    def _get_models_size(self):
        try:
            parameters = [i.detach().cpu().numpy() for i in self.model.parameters()]
            size = 0
            for i in range(len(parameters)-2):
                size += parameters[i].nbytes
            return int(size)
        except Exception as e:
            logger.error("_get_models_size error")
            logger.error("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))