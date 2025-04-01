import sys
import copy
import logging
import json
import pickle

import numpy as np

import flwr as fl

# from rando import local_concept_drift_config
from utils.models_utils import load_model, get_weights, load_data, set_weights, test, train
import torch

logging.basicConfig(level=logging.INFO)  # Configure logging
logger = logging.getLogger(__name__)  # Create logger for the module

def global_concept_drift_config(ME, n_rounds, alphas, experiment_id, seed=0):
    try:
        np.random.seed(seed)
        if experiment_id > 0:
            if experiment_id == 1:
                ME_concept_drift_rounds = [[int(n_rounds * 0.4), int(n_rounds * 0.8)], [int(n_rounds * 0.4), int(n_rounds * 0.8)]]
                new_alphas = [[10.0, 0.1], [0.1, 10.0]]

                config = {me: {"concept_drift_rounds": ME_concept_drift_rounds[me], "new_alphas": new_alphas[me]} for me in range(ME)}
            elif experiment_id == 3:
                ME_concept_drift_rounds = [[int(n_rounds * 0.2), int(n_rounds * 0.6)], [int(n_rounds * 0.3), int(n_rounds * 0.7)]]
                new_alphas = [[10.0, 0.1], [0.1, 10.0]]
            elif experiment_id == 4:
                ME_concept_drift_rounds = [[int(n_rounds * 0.2), int(n_rounds * 0.5), int(n_rounds * 0.8)],
                                           [int(n_rounds * 0.2), int(n_rounds * 0.5), int(n_rounds * 0.8)]]
                new_alphas = [[10.0, 1.0, 0.1], [0.1, 1.0, 10.0]]

            elif experiment_id == 5:
                ME_concept_drift_rounds = [[int(n_rounds * 0.2), int(n_rounds * 0.5), int(n_rounds * 0.8)],
                                           [int(n_rounds * 0.2), int(n_rounds * 0.5), int(n_rounds * 0.8)]]
                new_alphas = [[0.1, 1.0, 10.0], [10.0, 1.0, 0.1]]

            elif experiment_id == 6:
                # Melhor
                ME_concept_drift_rounds = [[int(n_rounds * 0.2), int(n_rounds * 0.5), int(n_rounds * 0.8)],
                                           [int(n_rounds * 0.2), int(n_rounds * 0.5), int(n_rounds * 0.8)]]
                new_alphas = [[0.1, 1.0, 10.0], [0.1, 1.0, 10.0]]

            elif experiment_id == 7:
                ME_concept_drift_rounds = [[int(n_rounds * 0.2), int(n_rounds * 0.5), int(n_rounds * 0.8)],
                                           [int(n_rounds * 0.2), int(n_rounds * 0.5), int(n_rounds * 0.8)]]
                new_alphas = [[10.0, 1.0, 0.1], [10.0, 1.0, 0.1]]

            elif experiment_id == 8:
                ME_concept_drift_rounds = [[int(n_rounds * 0.2), int(n_rounds * 0.5), int(n_rounds * 0.8)],
                                           [int(n_rounds * 0.2), int(n_rounds * 0.5), int(n_rounds * 0.8)]]
                new_alphas = [[10.0, 10.0, 10.0], [10.0, 10.0, 10.0]]


            config = {me: {"concept_drift_rounds": ME_concept_drift_rounds[me], "new_alphas": new_alphas[me]} for me in range(ME)}
        else:
            config = {}
        # else:
        #     config = {}
        return config

    except Exception as e:
        logger.error("global_concept_drift_config error")
        logger.error("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))

def local_concept_drift_config(ME, n_rounds, alphas, experiment_id, seed=0):
    try:
        np.random.seed(seed)
        if experiment_id > 0:
            if experiment_id == 2:
                n_concept_drifts = 10
                ME_concept_drift_rounds = [[] for me in range(ME)]
                for me in range(ME):
                    # ME_concept_drift_rounds[me] += np.random.choice([i for i in range(1, n_rounds + 1)], n_concept_drifts).tolist()
                    ME_concept_drift_rounds[me] += [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95]
                config = {me: {"concept_drift_rounds": ME_concept_drift_rounds[me], "new_alphas": [alphas[me]] * n_concept_drifts} for me in range(ME)}
        else:
            config = {}
        return config

    except Exception as e:
        logger.error("local_concept_drift_config error")
        logger.error("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))

class ClientMultiFedAvg(fl.client.NumPyClient):
    def __init__(self, args):
        try:
            self.args = args
            self.model = [load_model(args.model[me], args.dataset[me], args.strategy, args.device) for me in range(len(args.model))]
            self.alpha = [float(i) for i in args.alpha]
            self.initial_alpha = self.alpha
            self.ME = len(self.model)
            self.number_of_rounds = args.number_of_rounds
            logger.info("Preparing data...")
            logger.info("""args do cliente: {} {}""".format(self.args.client_id, self.alpha))
            self.client_id = args.client_id
            self.trainloader = [None] * self.ME
            self.recent_trainloader = [None] * self.ME
            self.valloader = [None] * self.ME
            self.optimizer = [None] * self.ME
            self.index = 0
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
            # Concept drift parameters
            self.concept_drift_experiment_id = self.args.concept_drift_experiment_id
            self.concept_drift_window = [0] * self.ME

            self.concept_drift_config = global_concept_drift_config(self.ME, self.number_of_rounds, self.alpha, self.concept_drift_experiment_id)
            logger.info(f"concept drift config {self.concept_drift_config} concept drift id {self.concept_drift_experiment_id}")
            for me in range(self.ME):
                self.trainloader[me], self.valloader[me] = load_data(
                    dataset_name=self.args.dataset[me],
                    alpha=self.alpha[me],
                    data_sampling_percentage=self.args.data_percentage,
                    partition_id=self.args.client_id,
                    num_partitions=self.args.total_clients + 1,
                    batch_size=self.args.batch_size,
                )
                self.recent_trainloader[me] = copy.deepcopy(self.trainloader[me])
                self.optimizer[me] = self._get_optimizer(dataset_name=self.args.dataset[me], me=me)
                logger.info("""leu dados cid: {} dataset: {} size:  {}""".format(self.args.client_id, self.args.dataset[me], len(self.trainloader[me].dataset)))

        except Exception as e:
            logger.error("__init__ error")
            logger.error("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))

    def fit(self, parameters, config):
        """Train the model with data of this client."""
        try:
            # logger.info("""fit cliente inicio config {} device {}""".format(config, self.device))
            t = config['t']
            me = config['me']
            self.lt[me] = t
            # Update alpha to simulate global concept drift
            alpha_me = self._get_current_alpha(t, me)
            if self.concept_drift_config != {}:
                if self.alpha[me] != alpha_me or (t in self.concept_drift_config[me]["concept_drift_rounds"] and self.concept_drift_experiment_id != 8):
                    self.alpha[me] = alpha_me
                    # self.index = {0: 1, 1: 2, 2: 0}[self.index]
                    # index = self.index
                    # if t in self.concept_drift_config[me]["concept_drift_rounds"] and self.concept_drift_experiment_id == 2:
                    #     # index = np.argwhere(np.array(self.concept_drift_config[me]["concept_drift_rounds"]) == t)[0][0] + 1
                    #     index = 0
                    index = 0
                    self.recent_trainloader[me], self.valloader[me] = load_data(
                        dataset_name=self.args.dataset[me],
                        alpha=self.alpha[me],
                        data_sampling_percentage=self.args.data_percentage,
                        partition_id=int((self.args.client_id + index) % self.args.total_clients),
                        num_partitions=self.args.total_clients + 1,
                        batch_size=self.args.batch_size,
                    )
                elif t in self.concept_drift_config[me]["concept_drift_rounds"] and self.concept_drift_experiment_id == 8:
                    self.concept_drift_window[me] += 1

            self.trainloader[me] = self.recent_trainloader[me]
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
                self.n_classes[me],
                self.concept_drift_window[me]
            )
            results["me"] = me
            results["client_id"] = self.client_id
            results["Model size"] = self.models_size[me]
            logger.info("fit cliente fim")
            return get_weights(self.model[me]), len(self.trainloader[me].dataset), results
        except Exception as e:
            logger.error("fit error")
            logger.error("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))

    def evaluate(self, parameters, config):
        """Evaluate the model on the data this client has."""
        try:
            logger.info("""eval cliente inicio""".format(config))
            t = config["t"]
            parameters = pickle.loads(config["parameters"])
            evaluate_models = json.loads(config["evaluate_models"])
            tuple_me = {}
            # logger.info("""modelos para cliente avaliar {} {} {}""".format(evaluate_models, type(parameters), parameters.keys()))
            for me in evaluate_models:
                me = int(me)
                # Update alpha to simulate global concept drift
                alpha_me = self._get_current_alpha(t, me)
                logger.info(f"config concept drift {self.concept_drift_config}")
                if self.concept_drift_config != {}:
                    if self.alpha[me] != alpha_me or (t in self.concept_drift_config[me][
                        "concept_drift_rounds"] and self.concept_drift_experiment_id != 8):
                        self.alpha[me] = alpha_me
                        # self.index = {0: 1, 1: 2, 2: 0}[self.index]
                        # index = self.index
                        # if t in self.concept_drift_config[me]["concept_drift_rounds"] and self.concept_drift_experiment_id == 2:
                        #     # index = np.argwhere(np.array(self.concept_drift_config[me]["concept_drift_rounds"]) == t)[0][0] + 1
                        #     index = 0
                        index = 0
                        self.recent_trainloader[me], self.valloader[me] = load_data(
                            dataset_name=self.args.dataset[me],
                            alpha=self.alpha[me],
                            data_sampling_percentage=self.args.data_percentage,
                            partition_id=int((self.args.client_id + index) % self.args.total_clients),
                            num_partitions=self.args.total_clients + 1,
                            batch_size=self.args.batch_size,
                        )
                    elif t in self.concept_drift_config[me][
                        "concept_drift_rounds"] and self.concept_drift_experiment_id == 8 and t - self.lt[me] > 0:
                        self.concept_drift_window[me] += 1
                me_str = str(me)
                nt = t - self.lt[me]
                parameters_me = parameters[me_str]
                set_weights(self.model[me], parameters_me)
                loss, metrics = test(self.model[me], self.valloader[me], self.device, self.client_id, t,
                                     self.args.dataset[me], self.n_classes[me], self.concept_drift_window[me])
                metrics["Model size"] = self.models_size[me]
                metrics["Dataset size"] = len(self.valloader[me].dataset)
                metrics["me"] = me
                metrics["Alpha"] = self.alpha[me]
                logger.info("""eval cliente fim {} {}""".format(metrics["me"], metrics))
                tuple_me[me_str] = pickle.dumps((loss, len(self.valloader[me].dataset), metrics))
            return loss, len(self.valloader[me].dataset), tuple_me
        except Exception as e:
            logger.error("evaluate error")
            logger.error("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))

    def _get_current_alpha(self, server_round, me):

        try:
            if self.concept_drift_experiment_id == 0:
                return self.alpha[me]
            else:
                config = self.concept_drift_config[me]
                alpha = None
                for i, round_ in enumerate(config["concept_drift_rounds"]):
                    if server_round >= round_:
                        alpha = config["new_alphas"][i]

                if alpha is None:
                    alpha = self.alpha[me]

                return alpha
        except Exception as e:
            logger.error(f"_get_current_alpha error {self.concept_drift_config}")
            logger.error("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))


    def _get_models_size(self):
        try:
            models_size = []
            for me in range(self.ME):
                parameters = [i.detach().cpu().numpy() for i in self.model[me].parameters()]
                size = 0
                for i in range(len(parameters)):
                    size += parameters[i].nbytes
                models_size.append(int(size))

            return models_size
        except Exception as e:
            logger.error("_get_models_size error")
            logger.error("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))

    def _get_optimizer(self, dataset_name, me):
        try:
            return {
                    'EMNIST': torch.optim.SGD(self.model[me].parameters(), lr=self.args.learning_rate, momentum=0.9),
                    'MNIST': torch.optim.SGD(self.model[me].parameters(), lr=self.args.learning_rate, momentum=0.9),
                    'CIFAR10': torch.optim.SGD(self.model[me].parameters(), lr=self.args.learning_rate, momentum=0.9),
                    'GTSRB': torch.optim.SGD(self.model[me].parameters(), lr=self.args.learning_rate, momentum=0.9),
                    'WISDM-W': torch.optim.RMSprop(self.model[me].parameters(), lr=0.001, momentum=0.9),
                    'WISDM-P': torch.optim.RMSprop(self.model[me].parameters(), lr=0.001, momentum=0.9),
                    'ImageNet100': torch.optim.SGD(self.model[me].parameters(), lr=self.args.learning_rate, momentum=0.9),
                    'ImageNet': torch.optim.SGD(self.model[me].parameters(), lr=0.1),
                    "ImageNet_v2": torch.optim.Adam(self.model[me].parameters(), lr=0.01),
                    "Gowalla": torch.optim.RMSprop(self.model[me].parameters(), lr=0.001, momentum=0.9)}[dataset_name]
        except Exception as e:
            logger.error("_get_optimizer error")
            logger.error("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))