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

from utils.models_utils import load_model, get_weights, load_data, set_weights, test_fedpredict, test, train

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

class ClientMultiFedAvgMultiFedPredict(ClientMultiFedAvg):
    def __init__(self, args):
        try:
            super(ClientMultiFedAvgMultiFedPredict, self).__init__(args)
            self.global_model = [None] * self.ME
            self.p_ME = [None] * self.ME
            self.fc_ME = [0] * self.ME
            self.il_ME = [0] * self.ME
            for me in range(self.ME):
                # Copy of randomly initialized parameters
                self.global_model[me] = copy.deepcopy(self.model[me])
            self.previous_alpha = self.alpha

            self.p_ME, self.fc_ME, self.il_ME = self._get_datasets_metrics(self.trainloader, self.ME, self.client_id, self.n_classes)
        except Exception as e:
            logger.error("__init__ error")
            logger.error("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))

    def fit(self, parameters, config):
        """Train the model with data of this client."""
        try:
            parameters, size, results = super().fit(parameters, config)
            self.p_ME, self.fc_ME, self.il_ME = self._get_datasets_metrics(self.trainloader, self.ME, self.client_id,
                                                                           self.n_classes)
            me = config['me']
            results["fc"] = self.fc_ME[me]
            results["il"] = self.il_ME[me]
            return parameters, size, results
        except Exception as e:
            logger.error("fit error")
            logger.error("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))

    def evaluate(self, parameters, config):
        """Evaluate the model on the data this client has."""
        try:
            # logger.info("""eval cliente inicio""".format(config))
            t = config["t"]
            parameters = pickle.loads(config["parameters"])
            evaluate_models = json.loads(config["evaluate_models"])
            tuple_me = {}
            for me in evaluate_models:
                me = int(me)
                me_str = str(me)
                alpha_me = self._get_current_alpha(t, me)
                # if self.alpha[me] != alpha_me or t in self.concept_drift_config[me]["concept_drift_rounds"]:
                #     self.alpha[me] = alpha_me
                #     index = 0
                #     if t in self.concept_drift_config[me][
                #         "concept_drift_rounds"] and self.concept_drift_experiment_id == 2:
                #         index = np.argwhere(np.array(self.concept_drift_config[me]["concept_drift_rounds"]) == t)[0][
                #                     0] + 1
                #     self.recent_trainloader[me], self.valloader[me] = load_data(
                #         dataset_name=self.args.dataset[me],
                #         alpha=self.alpha[me],
                #         data_sampling_percentage=self.args.data_percentage,
                #         partition_id=int((self.args.client_id + index) % self.args.total_clients),
                #         num_partitions=self.args.total_clients + 1,
                #         batch_size=self.args.batch_size,
                #     )
                #     p_ME, fc_ME, il_ME = self._get_datasets_metrics(self.trainloader, self.ME, self.client_id,
                #                                                     self.n_classes)
                # else:
                #     p_ME, fc_ME, il_ME = self.p_ME, self.fc_ME, self.il_ME
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
                similarity = 0.5
                combined_model = fedpredict_client_torch(local_model=self.model[me], global_model=self.global_model[me],
                                                         t=t, T=100, nt=nt, similarity=similarity, device=self.device)
                loss, metrics = test_fedpredict(combined_model, self.valloader[me], self.device, self.client_id, t,
                                                self.args.dataset[me], self.n_classes[me], similarity, p_ME[me])
                # loss, metrics = test(combined_model, self.valloader[me], self.device, self.client_id, t,
                #                                 self.args.dataset[me], self.n_classes[me])
                metrics["Model size"] = self.models_size[me]
                metrics["Dataset size"] = len(self.valloader[me].dataset)
                metrics["me"] = me
                metrics["Alpha"] = self.alpha[me]
                # logger.info("""eval cliente fim {} {} similaridade {}""".format(metrics["me"], metrics, similarity))
                tuple_me[me_str] = pickle.dumps((loss, len(self.valloader[me].dataset), metrics))
            return loss, len(self.valloader[me].dataset), tuple_me
        except Exception as e:
            logger.error("evaluate error")
            logger.error("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))

    def _get_datasets_metrics(self, trainloader, ME, client_id, n_classes):
        try:
            p_ME = []
            fc_ME = []
            il_ME = []
            rate_new_data = 0.5
            for me in range(ME):
                labels_me = []
                n_classes_me = n_classes[me]
                p_me = {i: 0 for i in range(n_classes_me)}
                size = len(trainloader[me])
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
            return p_ME, fc_ME, il_ME
        except Exception as e:
            logger.error("_get_datasets_metrics error")
            logger.error("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))

    def combine_dataloaders(self, dataloader1, dataloader2, proportion1=0.6, proportion2=0.4):
        # Extraindo os dados e rótulos dos dataloaders
        data1, labels1 = next(iter(dataloader1))
        data2, labels2 = next(iter(dataloader2))

        # Calculando o número de amostras para cada DataLoader
        num_samples1 = int(len(data1) * proportion1)
        num_samples2 = int(len(data2) * proportion2)

        # Selecionando as amostras do primeiro DataLoader (60%)
        selected_data1 = data1[:num_samples1]
        selected_labels1 = labels1[:num_samples1]

        # Selecionando as amostras do segundo DataLoader (40%)
        selected_data2 = data2[:num_samples2]
        selected_labels2 = labels2[:num_samples2]

        # Concatenando os dados e rótulos selecionados
        combined_data = torch.cat((selected_data1, selected_data2), dim=0)
        combined_labels = torch.cat((selected_labels1, selected_labels2), dim=0)

        # Criando um novo TensorDataset com os dados combinados
        combined_dataset = TensorDataset(combined_data, combined_labels)

        # Criando um DataLoader para os dados combinados
        combined_dataloader = DataLoader(combined_dataset, batch_size=32, shuffle=True)

        return combined_dataloader
