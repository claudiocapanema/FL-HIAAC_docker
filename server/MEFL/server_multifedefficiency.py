import sys
import pickle
import logging

import torch
import numpy as np
from typing import List, Tuple
from flwr.common import Metrics

from typing import Callable, Optional, Union

from flwr.common import (
    FitIns,
    EvaluateIns,
    EvaluateRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
)

from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy

import random
from server.MEFL.server_multifedavg import MultiFedAvg

from flwr.common.logger import log
from flwr.server.strategy.aggregate import aggregate, aggregate_inplace, weighted_loss_avg

from logging import WARNING

import sys
import logging

from typing import Callable, Optional

from flwr.common import (
    FitIns,
    FitRes,
    EvaluateIns,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    parameters_to_ndarrays,
    ndarrays_to_parameters,
    Scalar,
)
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy

import random
from server.MEFL.server_multifedavg import MultiFedAvg
import torch

# Initialize Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def ema(dados):

    try:
        # Definir o número de períodos (N)
        n = 5

        # Calcular o fator de suavização (alpha)
        alpha = 2 / (n + 1)

        # Inicializar o array para armazenar as EMAs
        ema = np.zeros_like(dados)

        # Calcular a primeira EMA (como a média simples dos primeiros N valores)
        ema[0] = np.mean(dados[:n])

        # Calcular as EMAs subsequentes
        for i in range(1, len(dados)):
            ema[i] = alpha * dados[i] + (1 - alpha) * ema[i - 1]

        return ema

        # Exibir os resultados
        print("Valores originais:", dados, len(dados))
        print("EMA:", ema, len(ema))

    except Exception as e:
        logger.error("ema error")
        logger.error("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))

def weighted_loss_avg(results: list[tuple[int, float]]) -> float:
    """Aggregate evaluation results obtained from multiple clients."""
    num_total_evaluation_examples = sum(num_examples for (num_examples, _) in results)
    weighted_losses = [num_examples * loss for num_examples, loss in results]
    return sum(weighted_losses) / num_total_evaluation_examples


# pylint: disable=line-too-long
class MultiFedEfficiency(MultiFedAvg):
    """MultiFedEfficiency strategy.

    Implementation based on

    Parameters
    ----------
    fraction_fit : float, optional
        Fraction of clients used during training. In case `min_fit_clients`
        is larger than `fraction_fit * available_clients`, `min_fit_clients`
        will still be sampled. Defaults to 1.0.
    fraction_evaluate : float, optional
        Fraction of clients used during validation. In case `min_evaluate_clients`
        is larger than `fraction_evaluate * available_clients`,
        `min_evaluate_clients` will still be sampled. Defaults to 1.0.
    min_fit_clients : int, optional
        Minimum number of clients used during training. Defaults to 2.
    min_evaluate_clients : int, optional
        Minimum number of clients used during validation. Defaults to 2.
    min_available_clients : int, optional
        Minimum number of total clients in the system. Defaults to 2.
    evaluate_fn : Optional[Callable[[int, NDArrays, Dict[str, Scalar]],Optional[Tuple[float, Dict[str, Scalar]]]]]
        Optional function used for validation. Defaults to None.
    on_fit_config_fn : Callable[[int], Dict[str, Scalar]], optional
        Function used to configure training. Defaults to None.
    on_evaluate_config_fn : Callable[[int], Dict[str, Scalar]], optional
        Function used to configure validation. Defaults to None.
    accept_failures : bool, optional
        Whether or not accept rounds containing failures. Defaults to True.
    initial_parameters : Parameters, optional
        Initial global model parameters.
    fit_metrics_aggregation_fn : Optional[MetricsAggregationFn]
        Metrics aggregation function, optional.
    evaluate_metrics_aggregation_fn : Optional[MetricsAggregationFn]
        Metrics aggregation function, optional.
    inplace : bool (default: True)
        Enable (True) or disable (False) in-place aggregation of model updates.
    """

    # pylint: disable=too-many-arguments,too-many-instance-attributes, line-too-long
    def __init__(
        self,
        *,
            args,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        evaluate_fn: Optional[
            Callable[
                [int, NDArrays, dict[str, Scalar]],
                Optional[tuple[float, dict[str, Scalar]]],
            ]
        ] = None,
        on_fit_config_fn: Optional[Callable[[int], dict[str, Scalar]]] = None,
        on_evaluate_config_fn: Optional[Callable[[int], dict[str, Scalar]]] = None,
        accept_failures: bool = True,
        initial_parameters: Optional[Parameters] = None,
        fit_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        inplace: bool = True,
    ) -> None:
        try:
            super().__init__(fraction_fit=fraction_fit, fraction_evaluate=fraction_evaluate,
                             min_fit_clients=min_fit_clients, min_evaluate_clients=min_evaluate_clients,
                             min_available_clients=min_available_clients, evaluate_fn=evaluate_fn,
                             on_fit_config_fn=on_fit_config_fn, on_evaluate_config_fn=on_evaluate_config_fn,
                             accept_failures=accept_failures, initial_parameters=initial_parameters,
                             fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
                             evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn, inplace=inplace,
                             args=args)
            self.heterogeneity_degree = {round_: [None] * self.ME for round_ in range(1, self.number_of_rounds + 1)}
            self.fc = {round_: [None] * self.ME for round_ in range(1, self.number_of_rounds + 1)}
            self.il = {round_: [None] * self.ME for round_ in range(1, self.number_of_rounds + 1)}
            self.similarity = {round_: [None] * self.ME for round_ in range(1, self.number_of_rounds + 1)}
            self.client_metrics = {
                cid: {me: {alpha: {"fc": None, "il": None, "similarity": None} for alpha in [0.1, 1.0, 10.0]} for me in
                      range(self.ME)} for cid in range(1, self.total_clients + 1)}
            self.min_training_clients_per_model = 3
            self.free_budget = int(
                self.fraction_fit * self.total_clients) - self.min_training_clients_per_model * self.ME
            self.ME_round_loss = {me: [] for me in range(self.ME)}
            self.checkpoint_models = {me: {} for me in range(self.ME)}
            self.round_initial_parameters = [None] * self.ME
            self.last_drift = 0
            self.min_training_clients_per_model = 3

            self.test_metrics_names = ["Accuracy", "Balanced accuracy", "Loss", "Round (t)", "Fraction fit",
                                       "# training clients", "training clients and models", "Model size", "Alpha", "fc",
                                       "il", "dh", "ps"]
            self.train_metrics_names = ["Accuracy", "Balanced accuracy", "Loss", "Round (t)", "Fraction fit",
                                        "# training clients", "training clients and models", "Model size", "Alpha"]
            self.rs_test_acc = {me: [] for me in range(self.ME)}
            self.rs_test_auc = {me: [] for me in range(self.ME)}
            self.rs_train_loss = {me: [] for me in range(self.ME)}
            self.results_train_metrics = {me: {metric: [] for metric in self.train_metrics_names} for me in
                                          range(self.ME)}
            self.results_train_metrics_w = {me: {metric: [] for metric in self.train_metrics_names} for me in
                                            range(self.ME)}
            self.results_test_metrics = {me: {metric: [] for metric in self.test_metrics_names} for me in
                                         range(self.ME)}
            self.results_test_metrics_w = {me: {metric: [] for metric in self.test_metrics_names} for me in
                                           range(self.ME)}
            self.clients_results_test_metrics = {me: {metric: [] for metric in self.test_metrics_names} for me in
                                                 range(self.ME)}
            self.loss_range = {me: [] for me in range(self.ME)}
            self.parameters_aggregated_mefl = {me: [] for me in range(self.ME)}
            self.metrics_aggregated_mefl = {me: [] for me in range(self.ME)}
        except Exception as e:
            logger.error("__init__ error")
            logger.error("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))

    def configure_fit(
            self, server_round: int, parameters: dict, client_manager: ClientManager
    ) -> list[tuple[ClientProxy, FitIns]]:
        try:
            torch.random.manual_seed(server_round)
            random.seed(server_round)
            np.random.seed(server_round)
            """Configure the next round of training."""
            config = {}
            if self.on_fit_config_fn is not None:
                # Custom fit config function provided
                config = self.on_fit_config_fn(server_round)

            # Sample clients
            logger.info("Waiting for available clients...")
            client_manager.wait_for(self.total_clients, 120)
            sample_size, min_num_clients = self.num_fit_clients(
                client_manager.num_available()
            )

            n_clients = int(self.total_clients * self.fraction_fit)

            logging.info(
                """sample clientes {} {} disponiveis {} rodada {} n clients {}""".format(sample_size,
                                                                                         min_num_clients,
                                                                                         client_manager.num_available(),
                                                                                         server_round, n_clients))
            clients = client_manager.sample(
                num_clients=n_clients, min_num_clients=n_clients
            )

            n = len(clients) // self.ME
            selected_clients_m = np.array_split(clients, self.ME)
            # training_intensity_me = [int((self.fraction_fit * self.total_clients) / self.ME)] * self.ME
            trained_models, training_intensity_me = self.random_selection(server_round, clients)

            selected_clients_m = []
            i = 0
            for me in range(self.ME):
                training_intensity = training_intensity_me[me]
                if training_intensity > 0:
                    logger.info(f"valo: {i} {training_intensity}")
                    j = i + training_intensity
                    selected_clients_m.append(clients[i:j])
                    i = j
                else:
                    selected_clients_m.append([])

            self.n_trained_clients = len(clients)
            logging.info("""selecionados {} por modelo {} rodada {}""".format(self.n_trained_clients,
                                                                              [len(i) for i in selected_clients_m],
                                                                              server_round))

            # Return client/config pairs
            clients_m = []
            for me in trained_models:
                sc = selected_clients_m[me]
                if server_round > 1:
                    self.round_initial_parameters[me] = parameters[me]
                for client in sc:
                    self.selected_clients_m_ids_random[me].append(client.cid)
                    config = {"t": server_round, "me": me}
                    if type(parameters) is dict:
                        fit_ins = FitIns(parameters[me], config)
                    else:
                        fit_ins = FitIns(parameters, config)
                    clients_m.append((client, fit_ins))
            return clients_m
        except Exception as e:
            logger.error("configure_fit error")
            logger.error("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))

    def aggregate_fit(
            self,
            server_round: int,
            results: list[tuple[ClientProxy, FitRes]],
            failures: list[Union[tuple[ClientProxy, FitRes], BaseException]],
    ) -> tuple[Optional[Parameters], dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""
        try:
            if not results:
                return None, {}
            # Do not aggregate if there are failures and failures are not accepted
            if not self.accept_failures and failures:
                return None, {}

            self.selected_clients_m = [[] for me in range(self.ME)]
            logger.info("no agrega")
            results_mefl = {me: [] for me in range(self.ME)}
            num_samples = {me: [] for me in range(self.ME)}
            fc = {me: [] for me in range(self.ME)}
            il = {me: [] for me in range(self.ME)}
            similarity = {me: [] for me in range(self.ME)}
            dh = {me: [] for me in range(self.ME)}
            trained_models = []
            for i in range(len(results)):
                _, result = results[i]
                me = result.metrics["me"]
                num_samples[me].append(result.num_examples)
                client_id = result.metrics["client_id"]
                fc[me].append(result.metrics["fc"])
                il[me].append(result.metrics["il"])
                similarity[me].append(result.metrics["similarity"])
                alpha = result.metrics["alpha"]
                self.client_metrics[client_id][me][alpha]["fc"] = result.metrics["fc"]
                self.client_metrics[client_id][me][alpha]["il"] = result.metrics["il"]
                self.client_metrics[client_id][me][alpha]["similarity"] = result.metrics["similarity"]
                s = result.metrics["similarity"]
                logger.info(f"similaridade do cliente {client_id} e {s} rodada {server_round}")

                self.selected_clients_m[me].append(client_id)
                results_mefl[me].append(results[i])
                if me not in trained_models:
                    trained_models.append(me)

            logger.info(f"antes fc {fc} il {il} rodada {server_round}")
            for me in trained_models:
                fc[me] = self._weighted_average(fc[me], num_samples[me])
                il[me] = self._weighted_average(il[me], num_samples[me])
                similarity[me] = self._weighted_average(similarity[me], num_samples[me])
                similarity[me] = 1 if similarity[me] >= 0.98 else similarity[me]
                logger.info(f"fc {fc} il {il}  heterogeneity degree {self.heterogeneity_degree[server_round]}")
                self.heterogeneity_degree[server_round][me] = (fc[me] + (1-il[me])) / 2
                self.fc[server_round][me] = fc[me]
                self.il[server_round][me] = il[me]
                self.similarity[server_round][me] = similarity[me]

            # logger.info(f"need {self.heterogeneity_degree[server_round]} rodada {server_round}")
            #
            # self.need_for_training = np.array(self.heterogeneity_degree[server_round]) / np.sum(self.heterogeneity_degree[server_round])
            # logger.info(f"need normalizado {self.need_for_training} rodada {server_round}")

            aggregated_ndarrays_mefl = {me: None for me in range(self.ME)}
            aggregated_ndarrays_mefl = {me: [] for me in range(self.ME)}
            weights_results_mefl = {me: [] for me in range(self.ME)}
            parameters_aggregated_mefl = {me: [] for me in range(self.ME)}

            if self.inplace:
                for me in trained_models:
                    # Does in-place weighted average of results
                    aggregated_ndarrays_mefl[me] = aggregate_inplace(results_mefl[me])
            else:
                for me in trained_models:
                    # Convert results
                    weights_results = [
                        (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
                        for _, fit_res in results_mefl[me]
                    ]
                    aggregated_ndarrays_mefl[me] = aggregate(weights_results)
                    if len(weights_results) > 1:
                        aggregated_ndarrays_mefl[me] = aggregate(weights_results)
                    elif len(weights_results) == 1:
                        aggregated_ndarrays_mefl[me] = results_mefl[me][1].parameters

            for me in trained_models:
                logger.info(f"tamanho para modelo {me} rodada {server_round} {len(aggregated_ndarrays_mefl[me])}")
                parameters_aggregated_mefl[me] = ndarrays_to_parameters(aggregated_ndarrays_mefl[me])

            # Aggregate custom metrics if aggregation fn was provided
            metrics_aggregated_mefl = {me: [] for me in trained_models}
            for me in trained_models:
                if self.fit_metrics_aggregation_fn:
                    fit_metrics = [(res.num_examples, res.metrics) for _, res in results_mefl[me]]
                    metrics_aggregated_mefl[me] = self.fit_metrics_aggregation_fn(fit_metrics)
                elif server_round == 1:  # Only log this warning once
                    log(WARNING, "No fit_metrics_aggregation_fn provided")

            # Get losses
            for me in trained_models:
                self.ME_round_loss[me].append(metrics_aggregated_mefl[me]["Loss"])

            logger.info("""finalizou aggregated fit""")
            logger.info(f"finalizou aggregated fit {server_round}")

            # self.calculate_pseudo_t(server_round, self.ME_round_loss[me])

            for me in trained_models:
                self.parameters_aggregated_mefl[me] = parameters_aggregated_mefl[me]
                self.metrics_aggregated_mefl[me] = metrics_aggregated_mefl[me]

            if server_round > 10:
                self._save_data_metrics()

            return self.parameters_aggregated_mefl, self.metrics_aggregated_mefl
        except Exception as e:
            logger.error("aggregate_fit error")
            logger.error("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))

    def configure_evaluate(
            self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> list[tuple[ClientProxy, EvaluateIns]]:
        """Configure the next round of evaluation."""
        # Do not configure federated evaluation if fraction eval is 0.
        try:
            if self.fraction_evaluate == 0.0:
                return []

            logger.info("""inicio configure evaluate {}""".format(type(parameters)))
            # Parameters and config
            config = {}
            if self.on_evaluate_config_fn is not None:
                # Custom evaluation config function provided
                config = self.on_evaluate_config_fn(server_round)

            dict_ME = {}
            me = 0
            if type(parameters) is dict:
                for key in parameters:
                    parameters_me = parameters_to_ndarrays(parameters[key])
                    dict_ME[str(key)] = parameters_me
                logger.info(f"parameters is dict round {server_round} ke {parameters.keys()}")
            else:
                logger.info(f"parameters is not dict round {server_round} ke {type(parameters)}")
            dict_ME = pickle.dumps(dict_ME)

            config["evaluate_models"] = str([me for me in range(self.ME)])
            config["t"] = server_round
            logger.info("""config evaluate antes {}""".format(config))
            config["parameters"] = dict_ME
            config["homogeneity_degree"] = pickle.dumps(self.heterogeneity_degree[server_round])
            config["fc"] = pickle.dumps(self.fc[server_round])
            config["il"] = pickle.dumps(self.il[server_round])
            config["similarity"] = pickle.dumps(self.similarity[server_round])
            evaluate_ins = EvaluateIns(parameters[0], config)

            # Sample clients
            sample_size, min_num_clients = self.num_evaluation_clients(
                client_manager.num_available()
            )

            clients = client_manager.sample(
                num_clients=sample_size, min_num_clients=min_num_clients
            )

            logger.info(f"final configure evaluate {server_round}")

            # Return client/config pairs
            return [(client, evaluate_ins) for client in clients]
        except Exception as e:
            logger.error("configure_evaluate error")
            logger.error("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))

    def add_metrics(self, server_round, metrics_aggregated, me):
        try:
            metrics_aggregated[me]["Fraction fit"] = self.fraction_fit
            metrics_aggregated[me]["# training clients"] = self.n_trained_clients
            metrics_aggregated[me]["training clients and models"] = self.selected_clients_m[me]
            metrics_aggregated[me]["fc"] = self.fc[server_round][me]
            metrics_aggregated[me]["il"] = self.il[server_round][me]
            metrics_aggregated[me]["ps"] = self.similarity[server_round][me]
            metrics_aggregated[me]["dh"] = self.heterogeneity_degree[server_round][me]

            for metric in metrics_aggregated[me]:
                self.results_test_metrics[me][metric].append(metrics_aggregated[me][metric])
        except Exception as e:
            logger.error("add_metrics error")
            logger.error("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))

    def calculate_pseudo_t(self, t, losses):

        try:
            ema_list = ema(losses)
            ema_value = ema_list[-1]
            logger.info(f"ema rodada {t} orignal {losses} ema {ema_list}")
            # Calcula a diferença absoluta entre cada elemento do vetor e x
            diferencas = np.abs(np.array(losses) - ema_value)
            # Retorna o índice da menor diferença
            return np.argmin(diferencas) + 1

        except Exception as e:
            logger.error("calculate_pseudo_t error")
            logger.error("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))

    def _save_data_metrics(self):

        try:
            for me in range(self.ME):
                algo = self.dataset[me] + "_" + self.strategy_name
                result_path = self.get_result_path("test")
                file_path = result_path + "{}_metrics.csv".format(algo)
                rows = []
                head = ["cid", "me", "Alpha", "fc", "il", "ps", "dh"]
                self._write_header(file_path, head, mode='w')
                for cid in range(1, self.total_clients + 1):
                    for alpha in [0.1, 1.0, 10.0]:
                        fc = self.client_metrics[cid][me][alpha]["fc"]
                        il = self.client_metrics[cid][me][alpha]["il"]
                        if fc is not None and il is not None:
                            dh = (fc + (1 - il)) / 2
                        else:
                            dh = None
                        row = [cid, me, alpha, fc, il, self.client_metrics[cid][me][alpha]["similarity"], dh]
                        rows.append(row)

                self._write_outputs(file_path, rows)

            logger.info(f"rows {rows}")

        except Exception as e:
            logger.error("_save_data_metrics error")
            logger.error("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))
            exit()

    def _weighted_average(self, values, weights):

        try:
            values = np.array([i * j for i, j in zip(values, weights)])
            values = np.sum(values) / np.sum(weights)
            return float(values)

        except Exception as e:
            logger.error("_weighted_average error")
            logger.error("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))

    # def process(self, t: int):
    #     """semi-convergence detection"""
    #     try:
    #         if t == 2:
    #             diff = self.tw_range[0] - self.tw_range[1]
    #             middle = self.tw_range[1] + diff // 2
    #             for me in range(self.ME):
    #                 # method 2
    #                 # r = diff * (1 - self.need_for_training[m])
    #                 # if self.need_for_training[m] >= 0.5:
    #                 #     # Makes it harder to reduce training intensity
    #                 #     lower = upper - t/2
    #                 #     upper = lower + diff * (1- self.need_for_training[m])
    #                 # else:
    #                 #     # Makes it easier to reduce training intensity
    #                 #     upper = self.tw_range[0]
    #                 #     lower = upper - diff * self.need_for_training[m]
    #
    #                 # method 1 funciona bem
    #                 lower = self.tw_range[1]
    #                 upper = self.tw_range[0]
    #                 r = diff * (1 - self.need_for_training[me])
    #                 # Smaller training reduction interval for higher need for training
    #                 lower = max(middle - r / 2, lower)
    #                 upper = min(middle + r / 2, upper)
    #
    #                 self.lim.append([upper, lower])
    #             self.lim[1] = [1, 1]
    #         flag = True
    #         logger.info(f"limites:  {self.lim}")
    #         # exit()
    #         loss_reduction = [0] * self.ME
    #         for me in range(self.ME):
    #             if not self.stop_cpd[me]:
    #                 self.rounds_since_last_semi_convergence[me] += 1
    #
    #                 """Stop CPD"""
    #                 logger.info(f"Modelo m:  {me}")
    #                 logger.info(f"tw:  {self.tw[me]}")
    #                 logger.info("""loss results test metrics {}""".format(self.results_test_metrics[me]["Loss"]))
    #                 losses = self.results_test_metrics[me]["Loss"][-(self.tw[me] + 1):]
    #                 losses = np.array([losses[i] - losses[i + 1] for i in range(len(losses) - 1)])
    #                 if len(losses) > 0:
    #                     loss_reduction[me] = losses[-1]
    #                 logger.info(f"Modelo {me}, losses: {losses}")
    #                 idxs = np.argwhere(losses < 0)
    #                 # lim = [[0.5, 0.25], [0.35, 0.15]]
    #                 logger.info("""tamanho lin {} ind me {}""".format(len(self.lim), me))
    #                 upper = self.lim[me][0]
    #                 lower = self.lim[me][1]
    #                 logger.info(f"Condição 1: {len(idxs) <= int(self.tw[me] * self.lim[me][0])}, Condição 2: {len(idxs) >= int(self.tw[me] * lower)}")
    #                 logger.info(f"{len(idxs)} {self.tw[me]} {upper} {lower} {int(self.tw[me] * upper)} {int(self.tw[me] * lower)}")
    #                 if self.rounds_since_last_semi_convergence[me] >= 4:
    #                     if len(idxs) <= int(self.tw[me] * upper) and len(idxs) >= int(self.tw[me] * lower):
    #                         self.rounds_since_last_semi_convergence[me] = 0
    #                         logger.info("a, remaining_clients_per_model, total_clientsb: ",
    #                               self.training_clients_per_model_per_round)
    #                         self.models_semi_convergence_rounds_n_clients[me].append({'round': t, 'n_training_clients':
    #                             self.training_clients_per_model_per_round[me][t]})
    #                         # more clients are trained for the semi converged model
    #                         logger.info(f"treinados na rodada passada: {me}, {self.training_clients_per_model_per_round[me][t - 2]}")
    #
    #                         if flag:
    #                             self.models_semi_convergence_flag[me] = True
    #                             self.models_semi_convergence_count[me] += 1
    #                             flag = False
    #
    #                     elif len(idxs) > int(self.tw[me] * upper):
    #                         self.rounds_since_last_semi_convergence[me] += 1
    #                         self.models_semi_convergence_count[me] -= 1
    #                         self.models_semi_convergence_count[me] = max(0, self.models_semi_convergence_count[me])
    #     except Exception as e:
    #         logger.error("process error")
    #         logger.error("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))

    def process(self, t: int):
        """semi-convergence detection"""
        try:
            flag = True
            logger.info(f"limites:  {self.lim}")
            # exit()
            loss_reduction = [0] * self.ME
            for me in range(self.ME):
                if not self.stop_cpd[me]:
                    self.rounds_since_last_semi_convergence[me] += 1

                    """Stop CPD"""
                    logger.info(f"Modelo m:  {me}")
                    logger.info(f"tw:  {self.tw[me]}")
                    logger.info("""loss results test metrics {}""".format(self.results_test_metrics[me]["Loss"]))
                    losses = self.ME_round_loss[me][self.loss_window:]
                    min_ = min(self.ME_round_loss[me])
                    max_ = max(self.ME_round_loss[me])

                    losses = np.array([losses[i] - losses[i + 1] for i in range(len(losses) - 1)])
                    if len(losses) > 0:
                        loss_reduction[me] = losses[-1]
                    logger.info(f"Modelo {me}, losses: {losses}")
                    idxs = np.argwhere(losses < 0)
                    # lim = [[0.5, 0.25], [0.35, 0.15]]
                    logger.info("""tamanho lin {} ind me {}""".format(len(self.lim), me))
                    upper = self.lim[me][0]
                    lower = self.lim[me][1]
                    logger.info(f"Condição 1: {len(idxs) <= int(self.tw[me] * self.lim[me][0])}, Condição 2: {len(idxs) >= int(self.tw[me] * lower)}")
                    logger.info(f"{len(idxs)} {self.tw[me]} {upper} {lower} {int(self.tw[me] * upper)} {int(self.tw[me] * lower)}")
                    if self.rounds_since_last_semi_convergence[me] >= 4:
                        if len(idxs) <= int(self.tw[me] * upper) and len(idxs) >= int(self.tw[me] * lower):
                            self.rounds_since_last_semi_convergence[me] = 0
                            logger.info("a, remaining_clients_per_model, total_clientsb: ",
                                  self.training_clients_per_model_per_round)
                            self.models_semi_convergence_rounds_n_clients[me].append({'round': t, 'n_training_clients':
                                self.training_clients_per_model_per_round[me][t]})
                            # more clients are trained for the semi converged model
                            logger.info(f"treinados na rodada passada: {me}, {self.training_clients_per_model_per_round[me][t - 2]}")

                            if flag:
                                self.models_semi_convergence_flag[me] = True
                                self.models_semi_convergence_count[me] += 1
                                flag = False

                        elif len(idxs) > int(self.tw[me] * upper):
                            self.rounds_since_last_semi_convergence[me] += 1
                            self.models_semi_convergence_count[me] -= 1
                            self.models_semi_convergence_count[me] = max(0, self.models_semi_convergence_count[me])
        except Exception as e:
            logger.error("process error")
            logger.error("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))

    def random_selection(self, t,  clients):

        try:
            g = torch.Generator()
            g.manual_seed(t)
            np.random.seed(t)
            random.seed(t)

            # if t > 1:
            #     budget = int((self.total_clients * self.fraction_fit) / self.ME)
            #     total_train_clients = int(self.fraction_fit * self.total_clients)
            #     selected_clients = np.random.choice(clients, total_train_clients, replace=False).tolist()
            #     selected_clients = [i for i in selected_clients]
            #
            #     selected_clients_m = [None] * self.ME
            #     cm_min = [self.min_training_clients_per_model] * self.ME
            #     cm = (self.need_for_training * total_train_clients).astype(int)
            #     cm = np.array([self.min_training_clients_per_model if i < self.min_training_clients_per_model else i for i in cm])
            #     free_budget = int(total_train_clients - np.sum(cm))
            #     k_nt = len(np.argwhere(cm > self.min_training_clients_per_model))
            #     if free_budget > 0 and k_nt > 0:
            #         free_budget_k = int(int(free_budget * 1) / k_nt)
            #         rest = free_budget - int(free_budget_k * k_nt)
            #
            #         logger.info(f"Free budget: {free_budget},  k nt: {k_nt}, Free budget k: {free_budget_k}, resto: {rest}")
            #
            #         for me in range(self.ME):
            #             if cm[me] > self.min_training_clients_per_model and cm[me] == budget:
            #                 cm[me] = int(cm[me] + free_budget_k)
            #                 if rest > 0:
            #                     cm[me] += 1
            #                     rest -= 1
            #                     rest = max(rest, 0)
            #
            #
            #     logger.info("""a : {}""".format(selected_clients_m))
            #     logger.info("""random: {}""".format(selected_clients))
            #     logger.info("""cm: {}""".format(cm))
            #     i = 0
            #     reverse_list = [k for k in range(self.ME)]
            #     for me in reverse_list:
            #         j = i + cm[me]
            #         selected_clients_m[me] = selected_clients[i: j]
            #         i = j
            #
            #     return cm
            # else:
            #     return [int((self.fraction_fit * self.total_clients) / self.ME)] * self.ME
            if self.experiment_id == 11:
                t_ref = 2
            elif self.experiment_id == 13:
                t_ref = 50
            elif self.experiment_id == 12:
                t_ref = 70
            elif self.experiment_id == 14:
                t_ref = 80
            elif self.experiment_id == 15:
                t_ref = 50
                if t > t_ref or t == 1:
                    trained_models = [i for i in range(self.ME)]
                    return trained_models, [int((self.fraction_fit * self.total_clients) / self.ME)] * self.ME
                else:
                    if t % 2 == 0:
                        trained_models = [0]
                        cm = [int(self.fraction_fit * self.total_clients), 0]
                        return trained_models, cm
                    else:
                        trained_models = [1]
                        cm = [0, int(self.fraction_fit * self.total_clients)]
                        return trained_models, cm
            if t < t_ref:
                trained_models = [i for i in range(self.ME)]
                return trained_models, [int((self.fraction_fit * self.total_clients) / self.ME)] * self.ME
            else:
                if t % 2 == 0:
                    trained_models = [0]
                    cm = [int(self.fraction_fit * self.total_clients), 0]
                    return trained_models, cm
                else:
                    trained_models = [1]
                    cm = [0, int(self.fraction_fit * self.total_clients)]
                    return trained_models, cm

        except Exception as e:
            logger.error("Error random selection")
            logger.error('Error on line {} {} {}'.format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))