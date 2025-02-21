import logging
import os
from torch.nn.parameter import Parameter

from model.model import test, train
import torch

logging.basicConfig(level=logging.INFO)  # Configure logging
logger = logging.getLogger(__name__)  # Create logger for the module

from clients.client_fedavg import Client

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
        super().__init__(args)

    def fit(self, parameters, config):
        """Train the model with data of this client."""

        logger.info("""fit cliente inicio config {} device {}""".format(config, self.device))
        t = config['t']
        self.lt = t - self.lt
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
        for i in range(len(parameters)-2):
            size += parameters[i].nbytes
        return int(size)