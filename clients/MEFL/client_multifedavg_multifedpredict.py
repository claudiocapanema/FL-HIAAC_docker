import sys
import copy
import json
import pickle
import logging
import numpy as np
import torch

from clients.MEFL.client_multifedavg import ClientMultiFedAvg
from fedpredict import fedpredict_client_torch
# from fedpredict.utils.utils import cosine_similarity
from numpy.linalg import norm
from sklearn.cluster import KMeans

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

def aggregate_p(p_new, p_old, mean_similarity):

    # compute cosine similarity
    try:
        p = p_new * mean_similarity + p_old * (1 - mean_similarity)
        return p / np.sum(p)
    except Exception as e:
        logger.error("aggregate_p error")
        logger.error("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))

def calcular_media_intervalos(lista, valor):
    try:
        # Variáveis para armazenar a soma e a quantidade de intervalos
        soma_intervalos = 0
        quantidade_intervalos = 0

        # Variável para controlar o intervalo atual de posições consecutivas sem o valor
        contador = 0
        dentro_do_intervalo = False

        for i in range(len(lista)):
            if lista[i] == valor:
                if dentro_do_intervalo:
                    soma_intervalos += contador
                    quantidade_intervalos += 1
                    dentro_do_intervalo = False
                contador = 0
            else:
                contador += 1
                dentro_do_intervalo = True

        # Se o último intervalo terminar no final da lista
        if dentro_do_intervalo:
            soma_intervalos += contador
            quantidade_intervalos += 1

        # Calcular a média
        if quantidade_intervalos > 0:
            media = soma_intervalos / quantidade_intervalos
        else:
            media = 0

        return media
    except Exception as e:
        logger.error(f"media intervalos error")
        logger.error("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))

def mean_p(p_ME_list, ME, NT, t):

    # compute cosine similarity
    try:
        p_ME_average = [None] * ME
        for me in range(ME):
            nt = NT[me]
            if len(p_ME_list[me]) > 0:
                if nt == 100:
                    p_ME_average[me] = p_ME_list[me][-1]
                else:
                    # weight_i_list = []
                    # size = len(p_ME_list[me])
                    # for i in range(size):
                    #     weight = pow(nt, (size - i))
                    #     weight_i_list.append(1 - weight)
                    #
                    # weight_i_list = np.array(weight_i_list) / np.sum(weight_i_list)
                    #
                    # p_ME_average[me] = np.sum(p_ME_list[me] * weight_i_list, axis=0)

                    p_ME_average[me] = np.sum(p_ME_list[me], axis=0)
                    p_ME_average[me] = p_ME_average[me] / np.sum(p_ME_average[me])
                    # k = 3
                    # if len(p_ME_list[me]) % k == 0:
                    #
                    #     kmeans = KMeans(n_clusters=k, random_state=0)
                    #
                    #     # Ajuste do modelo aos dados
                    #     kmeans.fit(np.array(p_ME_list[me]))
                    #
                    #     # Obter os centróides (centros dos clusters)
                    #     centroides = kmeans.cluster_centers_
                    #
                    #     # Obter as etiquetas (que cluster cada ponto pertence)
                    #     etiquetas = kmeans.labels_
                    #     concept_drift_period = {i: 1 for i in range(k)}
                    #     for label_ in concept_drift_period:
                    #         concept_drift_period[label_] = calcular_media_intervalos(etiquetas, label_)
                    #     round_labels = [0] * 100
                    #     count = 0
                    #     label_ = 0
                    #     for i in range(len(round_labels)):
                    #
                    #         round_labels[i] = label_
                    #         count += 1
                    #         if count == int(concept_drift_period[label_] * 3.333):
                    #             label_ += 1
                    #             count = 0
                    #         if label_ == k:
                    #             label_ = 0
                    #     ground_true = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                    #     correct = 0
                    #     for i in range(len(round_labels)):
                    #         if round_labels[i] == ground_true[i] and i in [9, 19, 29, 39, 49, 59, 69, 79, 89]:
                    #             correct += 1
                    #
                    #     logger.info(f"concept_drift_period acertos {correct/9}")
                    #     logger.info(f"roundlab {round_labels}")



        return p_ME_average
    except Exception as e:
        logger.error(f"mean_p error {p_ME_list}")
        logger.error("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))

class ClientMultiFedAvgMultiFedPredict(ClientMultiFedAvg):
    def __init__(self, args):
        try:
            super(ClientMultiFedAvgMultiFedPredict, self).__init__(args)
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

            self.p_ME, self.fc_ME, self.il_ME = self._get_datasets_metrics(self.trainloader, self.ME, self.client_id, self.n_classes)
        except Exception as e:
            logger.error("__init__ error")
            logger.error("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))

    def fit(self, parameters, config):
        """Train the model with data of this client."""
        try:
            parameters, size, results = super().fit(parameters, config)
            p_ME, fc_ME, il_ME = self._get_datasets_metrics(self.trainloader, self.ME, self.client_id,
                                                                           self.n_classes)
            # for me in range(self.ME):
                # if len(self.p_ME_list[me]) > 0:
                #     if self.p_ME_list[me][-1] != self.p_ME[me]:
                #         self.p_ME_list[me].append(self.p_ME[me])
                #     elif len(self.p_ME_list[me]) == 0:
                #         self.p_ME_list[me].append(self.p_ME[me])
            me = config['me']
            t = config['t']
            self.p_ME[me] = p_ME[me]
            self.fc_ME[me] = fc_ME[me]
            self.il_ME[me] = il_ME[me]
            self.p_ME_list[me].append(self.p_ME[me])

            self.NT[me] = 0
            self.mean_p_ME = mean_p(self.p_ME_list, self.ME, self.NT, t)
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
            homogeneity_degree = pickle.loads(config["homogeneity_degree"])
            fc = pickle.loads(config["fc"])
            il = pickle.loads(config["il"])
            tuple_me = {}
            for me in range(self.ME):
                self.NT[me] = t - self.lt[me]
            for me in evaluate_models:
                me = int(me)
                me_str = str(me)
                alpha_me = self._get_current_alpha(t, me)
                # Comment to simulate the `Delayed labeling`
                self.trainloader[me] = self.recent_trainloader[me]
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
                        p_ME, fc_ME, il_ME = self._get_datasets_metrics(self.trainloader, self.ME, self.client_id,
                                                                        self.n_classes, self.concept_drift_window)
                    else:
                        p_ME, fc_ME, il_ME = self.p_ME, self.fc_ME, self.il_ME
                else:
                    p_ME, fc_ME, il_ME = self.p_ME, self.fc_ME, self.il_ME
                # if self.alpha[me] != alpha_me:
                #     self.alpha[me] = alpha_me
                #     self.trainloader[me], self.valloader[me] = load_data(
                #         dataset_name=self.args.dataset[me],
                #         alpha=self.alpha[me],
                #         data_sampling_percentage=self.args.data_percentage,
                #         partition_id=self.args.client_id,
                #         num_partitions=self.args.total_clients + 1,
                #         batch_size=self.args.batch_size,
                #     )
                #     p_ME, fc_ME, il_ME = self._get_datasets_metrics(self.trainloader, self.ME, self.client_id,
                #                                                     self.n_classes)
                # else:
                #     p_ME, fc_ME, il_ME = self.p_ME, self.fc_ME, self.il_ME
                nt = t - self.lt[me]
                parameters_me = parameters[me_str]
                set_weights(self.global_model[me], parameters_me)
                similarity = cosine_similarity(self.p_ME[me], p_ME[me])
                if fc[me] >= 0.97 and il[me] < 59:
                    similarity = 0
                combined_model = fedpredict_client_torch(local_model=self.model[me], global_model=self.global_model[me],
                                                         t=t, T=100, nt=nt, similarity=similarity, device=self.device)
                if fc[me] >= 0.97 and il[me] < 0.59:
                    similarity = 1
                    combined_model = self.global_model[me]
                loss, metrics = test_fedpredict(combined_model, self.valloader[me], self.device, self.client_id, t,
                                                self.args.dataset[me], self.n_classes[me], similarity, p_ME[me], self.concept_drift_window[me])
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
