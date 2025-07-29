import logging
import os
import sys

from clients.FL.client_fedavg import Client

from utils.models_utils import get_weights_fedkd, set_weights_fedkd, test_fedkd, train_fedkd, load_model
import torch
from fedpredict.utils.compression_methods.parameters_svd import parameter_svd_write, inverse_parameter_svd_reading
from fedpredict.utils.compression_methods.fedkd import fedkd_compression
from fedpredict.fedpredict_core import layer_compression_range

logging.basicConfig(level=logging.INFO)  # Configure logging
logger = logging.getLogger(__name__)  # Create logger for the module

# Make TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

class ClientFedKD(Client):
    def __init__(self, args):
        try:
            super().__init__(args)
            logger.info("Initializing ClientFedKD")
            self.lr_loss = torch.nn.MSELoss()
            self.round_of_last_fit = 0
            self.rounds_of_fit = 0
            self.accuracy_of_last_round_of_fit = 0
            self.start_server = 0
            # self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9)
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=3e-6)
            feature_dim = 512
            self.W_h = torch.nn.Linear(feature_dim, feature_dim, bias=False)
            self.MSE = torch.nn.MSELoss()
            model_shape = get_weights_fedkd(load_model(args.model[0], args.dataset[0], args.strategy, args.device))
            self.model_shape = [i.shape for i in model_shape]
            self.layers_compression_range = layer_compression_range(self.model_shape)
        except Exception as e:
            logger.error("__init__ error")
            logger.error("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))

    def fit(self, parameters, config):
        """Train the utils with data of this client."""
        try:
            logger.info("""fit cliente fedkd inicio config {} device {}""".format(config, self.device))
            t = config['t']
            self.model.to(self.device)
            # if t > 1:
            #     parameters = inverse_parameter_svd_reading(parameters, [i.detach().cpu().numpy().shape for i in
            #                                                             self.model.student.parameters()])
            if len(parameters) > 0:
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
                self.dataset,
                self.n_classes
            )
            self.models_size = self._get_models_size(parameters)
            results["Model size"] = self.models_size
            logger.info(f"fim fedkd client fit {results}")
            return get_weights_fedkd(self.model), len(self.trainloader.dataset), results

        except Exception as e:
            logger.error("fit")
            logger.error('Error on line {} {} {}'.format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))


    def evaluate(self, parameters, config):
        """Evaluate the utils on the data this client has."""
        try:
            logger.info("""eval cliente inicio""".format(config))
            t = config["t"]
            nt = t - self.lt
            set_weights_fedkd(self.model, parameters)
            loss, metrics = test_fedkd(self.model, self.valloader, self.device, self.client_id, t, self.dataset, self.n_classes)
            self.models_size = self._get_models_size(parameters)
            metrics["Model size"] = self.models_size
            metrics["Alpha"] = self.alpha
            metrics["nt"] = nt
            logger.info("eval cliente fim")
            return loss, len(self.valloader.dataset), metrics
        except Exception as e:
            logger.error("evaluate")
            logger.error('Error on line {} {} {}'.format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))

    def compress(self, server_round, parameters):

        try:
            layers_compression_range = self.layer_compression_range([i.shape for i in parameters])
            n_components_list = []
            for i in range(len(parameters)):
                compression_range = layers_compression_range[i]
                if compression_range > 0:
                    frac = 1 - server_round / self.n_rounds
                    compression_range = max(round(frac * compression_range), 1)
                else:
                    compression_range = None
                n_components_list.append(compression_range)

            parameters_to_send = parameter_svd_write(parameters, n_components_list)
            return parameters_to_send

        except Exception as e:
            print("compress")
            print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)