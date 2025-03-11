import logging
import json
import pickle

import flwr as fl

from utils.models_utils import load_model, get_weights, load_data, set_weights, test, train
import torch

logging.basicConfig(level=logging.INFO)  # Configure logging
logger = logging.getLogger(__name__)  # Create logger for the module

class ClientMultiFedAvg(fl.client.NumPyClient):
    def __init__(self, args):
        self.args = args
        self.model = [load_model(args.model[me], args.dataset[me], args.strategy, args.device) for me in range(len(args.model))]
        self.alpha = [float(i) for i in args.alpha]
        self.ME = len(self.model)
        logger.info("Preparing data...")
        logger.info("""args do cliente: {} {}""".format(self.args.client_id, self.alpha))
        self.client_id = args.client_id
        self.trainloader = [None] * self.ME
        self.valloader = [None] * self.ME
        self.optimizer = [None] * self.ME
        for me in range(self.ME):
            self.trainloader[me], self.valloader[me] = load_data(
                dataset_name=self.args.dataset[me],
                alpha=self.alpha[me],
                data_sampling_percentage=self.args.data_percentage,
                partition_id=self.args.client_id,
                num_partitions=self.args.total_clients + 1,
                batch_size=self.args.batch_size,
            )
            self.optimizer[me] = self._get_optimizer(dataset_name=self.args.dataset[me], me=me)
            logger.info("""leu dados cid: {} dataset: {} size:  {}""".format(self.args.client_id, self.args.dataset[me], len(self.trainloader[me].dataset)))

        self.local_epochs = self.args.local_epochs
        self.lr = self.args.learning_rate
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.lt = [0] * self.ME
        logger.info("ler model size")
        self.models_size = self._get_models_size()
        logger.info("leu model size")
        self.n_classes = [
            {'EMNIST': 47, 'MNIST': 10, 'CIFAR10': 10, 'GTSRB': 43, 'WISDM-W': 12, 'WISDM-P': 12, 'ImageNet': 15,
             "ImageNet_v2": 15, "Gowalla": 7}[dataset] for dataset in
            self.args.dataset]

    def fit(self, parameters, config):
        """Train the model with data of this client."""

        logger.info("""fit cliente inicio config {} device {}""".format(config, self.device))
        t = config['t']
        me = config['me']
        self.lt[me] = t - self.lt[me]
        if len(parameters) > 0:
            set_weights(self.model[me], parameters)
        self.optimizer[me] = self._get_optimizer(dataset_name=self.args.dataset[me], me=me)
        results = train(
            self.model[me],
            self.trainloader[me],
            self.valloader[me],
            self.optimizer[me],
            self.local_epochs,
            self.lr,
            self.device,
            self.client_id,
            t,
            self.args.dataset[me],
            self.n_classes[me]
        )
        results["me"] = me
        results["client_id"] = self.client_id
        logger.info("fit cliente fim")
        return get_weights(self.model[me]), len(self.trainloader[me].dataset), results

    def evaluate(self, parameters, config):
        """Evaluate the model on the data this client has."""
        logger.info("""eval cliente inicio""".format(config))
        t = config["t"]
        parameters = pickle.loads(config["parameters"])
        evaluate_models = json.loads(config["evaluate_models"])
        tuple_me = {}
        logger.info("""modelos para cliente avaliar {} {} {}""".format(evaluate_models, type(parameters), parameters.keys()))
        for me in evaluate_models:
            me = int(me)
            me_str = str(me)
            nt = t - self.lt[me]
            parameters_me = parameters[me_str]
            set_weights(self.model[me], parameters_me)
            loss, metrics = test(self.model[me], self.valloader[me], self.device, self.client_id, t, self.args.dataset[me], self.n_classes[me])
            metrics["Model size"] = self.models_size[me]
            metrics["Dataset size"] = len(self.valloader[me].dataset)
            metrics["me"] = me
            logger.info("""eval cliente fim {} {}""".format(metrics["me"], metrics))
            tuple_me[me_str] = pickle.dumps((loss, len(self.valloader[me].dataset), metrics))
        return loss, len(self.valloader[me].dataset), tuple_me

    def _get_models_size(self):
        models_size = []
        for me in range(self.ME):
            parameters = [i.detach().cpu().numpy() for i in self.model[me].parameters()]
            size = 0
            for i in range(len(parameters)):
                size += parameters[i].nbytes
            models_size.append(int(size))

        return models_size

    def _get_optimizer(self, dataset_name, me):

        return {
                'EMNIST': torch.optim.SGD(self.model[me].parameters(), lr=self.args.learning_rate, momentum=0.9),
                'MNIST': torch.optim.SGD(self.model[me].parameters(), lr=self.args.learning_rate, momentum=0.9),
                'CIFAR10': torch.optim.SGD(self.model[me].parameters(), lr=self.args.learning_rate, momentum=0.9),
                'GTSRB': torch.optim.SGD(self.model[me].parameters(), lr=self.args.learning_rate, momentum=0.9),
                'WISDM-W': torch.optim.SGD(self.model[me].parameters(), lr=self.args.learning_rate, momentum=0.9),
                'WISDM-P': torch.optim.SGD(self.model[me].parameters(), lr=self.args.learning_rate, momentum=0.9),
                'ImageNet100': torch.optim.SGD(self.model[me].parameters(), lr=self.args.learning_rate, momentum=0.9),
                'ImageNet': torch.optim.Adam(self.model[me].parameters(), lr=0.01),
                "ImageNet_v2": torch.optim.Adam(self.model[me].parameters(), lr=0.01),
                "Gowalla": torch.optim.SGD(self.model[me].parameters(), lr=self.args.learning_rate, momentum=0.9)}[dataset_name]