import sys
import copy
import json
import pickle
import logging
import numpy as np

from clients.MEFL.client_multifedavg import ClientMultiFedAvg
from fedpredict import fedpredict_client_torch
# from fedpredict.utils.utils import cosine_similarity
from numpy.linalg import norm

from utils.models_utils import load_model, get_weights, load_data, set_weights, test, train, test_fedpredict

logging.basicConfig(level=logging.INFO)  # Configure logging
logger = logging.getLogger(__name__)  # Create logger for the module

def cosine_similarity(p_1, p_2):

    # compute cosine similarity
    try:
        p_1_size = np.array(p_1).shape
        p_2_size = np.array(p_2).shape
        if p_1_size != p_2_size:
            raise Exception(f"Input sizes have different shapes: {p_1_size} and {p_2_size}. Please check your input data.")

        return np.dot(p_1, p_2) / (norm(p_1) * norm(p_2))
    except Exception as e:
        logger.error("cosine_similairty error")
        logger.error("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))

class ClientMultiFedAvgFedPredictDynamic(ClientMultiFedAvg):
    def __init__(self, args):
        super(ClientMultiFedAvgFedPredictDynamic, self).__init__(args)
        self.global_model = [None] * self.ME
        self.p_ME = [None] * self.ME
        for me in range(self.ME):
            # Copy of randomly initialized parameters
            self.global_model[me] = copy.deepcopy(self.model[me])
        self.previous_alpha = self.alpha

        self.p_ME, self.fc_ME, self.il_ME = self._get_datasets_metrics(self.trainloader, self.ME, self.client_id, self.n_classes)

    def fit(self, parameters, config):
        """Train the model with data of this client."""
        try:
            self.p_ME, self.fc_ME, self.il_ME = self._get_datasets_metrics(self.trainloader, self.ME, self.client_id, self.n_classes)
            parameters, size, results = super().fit(parameters, config)
            me = config['me']
            return parameters, size, results
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
            logger.info("""modelos para cliente avaliar {} {} {}""".format(evaluate_models, type(parameters), parameters.keys()))
            for me in evaluate_models:
                me = int(me)
                me_str = str(me)
                alpha_me = self._get_current_alpha(t, me)
                if self.alpha[me] != alpha_me:
                    self.alpha[me] = alpha_me
                    self.trainloader[me], self.valloader[me] = load_data(
                        dataset_name=self.args.dataset[me],
                        alpha=self.alpha[me],
                        data_sampling_percentage=self.args.data_percentage,
                        partition_id=self.args.client_id,
                        num_partitions=self.args.total_clients + 1,
                        batch_size=self.args.batch_size,
                    )
                    p_ME, fc_ME, il_ME = self._get_datasets_metrics(self.trainloader, self.ME, self.client_id,
                                                                    self.n_classes)
                else:
                    p_ME, fc_ME, il_ME = self.p_ME, self.fc_ME, self.il_ME
                nt = t - self.lt[me]
                parameters_me = parameters[me_str]
                set_weights(self.global_model[me], parameters_me)
                similarity = cosine_similarity(self.p_ME[me], p_ME[me])
                combined_model = fedpredict_client_torch(local_model=self.model[me], global_model=self.global_model[me],
                                                         t=t, T=100, nt=nt, similarity=similarity, device=self.device)
                loss, metrics = test_fedpredict(combined_model, self.valloader[me], self.device, self.client_id, t, self.args.dataset[me], self.n_classes[me], similarity, p_ME[me])
                metrics["Model size"] = self.models_size[me]
                metrics["Dataset size"] = len(self.valloader[me].dataset)
                metrics["me"] = me
                metrics["Alpha"] = self.alpha[me]
                logger.info("""eval cliente fim {} {} similaridade {}""".format(metrics["me"], metrics, similarity))
                tuple_me[me_str] = pickle.dumps((loss, len(self.valloader[me].dataset), metrics))
            return loss, len(self.valloader[me].dataset), tuple_me
        except Exception as e:
            logger.error("evaluate error")
            logger.error("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))

    def _get_datasets_metrics(self, trainloader, ME, client_id, n_classes):

        p_ME = []
        fc_ME = []
        il_ME = []
        for me in range(ME):
            labels_me = []
            n_classes_me = n_classes[me]
            p_me = {i: 0 for i in range(n_classes_me)}
            for batch in trainloader[me]:
                labels = batch["label"]
                labels_me += labels.detach().cpu().numpy().tolist()
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
