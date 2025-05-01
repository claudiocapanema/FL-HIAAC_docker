import sys
import copy
import logging
import json
import pickle
from fedpredict import fedpredict_client_torch
from utils.models_utils import load_model, get_weights, load_data, set_weights, test, train

import numpy as np
import torch
from clients.MEFL.client_multifedavg import ClientMultiFedAvg

logging.basicConfig(level=logging.INFO)  # Configure logging
logger = logging.getLogger(__name__)  # Create logger for the module

class ClientMultiFedEfficiency(ClientMultiFedAvg):
    def __init__(self, args):
        try:
            super().__init__(args)
            self.global_model = [None] * self.ME
            self.p_ME = [None] * self.ME
            self.p_ME_list = {me: [] for me in range(self.ME)}
            self.fc_ME = [0] * self.ME
            self.il_ME = [0] * self.ME
            self.similarity_ME = [[]] * self.ME
            self.mean_p_ME = [None] * self.ME
            self.NT = [None] * self.ME
            for me in range(self.ME):
                # Copy of randomly initialized parameters
                self.global_model[me] = copy.deepcopy(self.model[me])
            self.previous_alpha = self.alpha

            self.p_ME, self.fc_ME, self.il_ME = self._get_datasets_metrics(self.trainloader, self.ME, self.client_id,
                                                                           self.n_classes)
        except Exception as e:
            logger.error("__init__ error")
            logger.error("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))

    def fit(self, parameters, config):
        """Train the model with data of this client."""
        try:
            p_old = self.p_ME
            me = config['me']
            parameters, size, results = super().fit(parameters, config)
            p_ME, fc_ME, il_ME = self._get_datasets_metrics(self.trainloader, self.ME, self.client_id,
                                                                           self.n_classes, self.concept_drift_window)
            self.p_ME, self.fc_ME, self.il_ME = p_ME, fc_ME, il_ME

            t = config['t']
            self.p_ME[me] = p_ME[me]
            self.fc_ME[me] = fc_ME[me]
            self.il_ME[me] = il_ME[me]

            me = config['me']
            results["fc"] = self.fc_ME[me]
            results["il"] = self.il_ME[me]
            results["similarity"] = 0
            results["alpha"] = self.alpha[me]
            return parameters, size, results
        except Exception as e:
            logger.error("fit error")
            logger.error("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))

    def evaluate(self, parameters, config):
        """Train the model with data of this client."""
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
                        "concept_drift_rounds"] and self.concept_drift_config[me]["type"] in ["label_shift"]):
                        self.alpha[me] = alpha_me
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
                        "concept_drift_rounds"] and self.concept_drift_config[me]["type"] in ["concept_drift"] and t - \
                            self.lt[me] > 0:
                        self.concept_drift_window[me] += 1
                me_str = str(me)
                nt = t - self.lt[me]
                parameters_me = parameters[me_str]
                set_weights(self.global_model[me], parameters_me)
                # combined_model = fedpredict_client_torch(local_model=self.model[me], global_model=self.global_model[me],
                #                                          t=t, T=100, nt=nt, device=self.device, logs=True)
                loss, metrics = test(self.global_model[me], self.valloader[me], self.device, self.client_id, t,
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

    def _get_datasets_metrics(self, trainloader, ME, client_id, n_classes, concept_drift_window=None):

        try:
            p_ME = []
            fc_ME = []
            il_ME = []
            for me in range(ME):
                labels_me = []
                n_classes_me = n_classes[me]
                p_me = {i: 0 for i in range(n_classes_me)}
                with (torch.no_grad()):
                    for batch in trainloader[me]:
                        labels = batch["label"]
                        labels = labels.to("cuda:0")

                        if concept_drift_window is not None:
                            labels = (labels + concept_drift_window[me])
                            labels = labels % n_classes[me]
                        labels = labels.detach().cpu().numpy()
                        labels_me += labels.tolist()
                    unique, count = np.unique(labels_me, return_counts=True)
                    data_unique_count_dict = dict(zip(np.array(unique).tolist(), np.array(count).tolist()))
                    for label in data_unique_count_dict:
                        p_me[label] = data_unique_count_dict[label]
                    p_me = np.array(list(p_me.values()))
                    fc_me = len(np.argwhere(p_me > 0)) / n_classes_me
                    il_me = len(np.argwhere(p_me < np.sum(p_me) / n_classes_me)) / n_classes_me
                    p_me = p_me / np.sum(p_me)
                    p_ME.append(p_me)
                    fc_ME.append(fc_me)
                    il_ME.append(il_me)
                    logger.info(f"p_me {p_me} fc_me {fc_me} il_me {il_me} model {me} client {client_id}")
            return p_ME, fc_ME, il_ME
        except Exception as e:
            logger.error("_get_datasets_metrics error")
            logger.error("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))