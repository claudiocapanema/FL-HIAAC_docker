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

# Initialize Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
            self.tw = []
            for d in self.dataset:
                self.tw.append({'EMNIST': args.tw, 'MNIST': args.tw, 'GTSRB': args.tw, "WISDM-W": args.tw, "WISDM-P": args.tw, "ImageNet": args.tw, "CIFAR10": args.tw,
                                "ImageNet_v2": args.tw, "Gowalla": args.tw}[d])
            self.n_classes = [
                {'EMNIST': 47, 'MNIST': 10, 'CIFAR10': 10, 'GTSRB': 43, 'WISDM-W': 12, 'WISDM-P': 12, 'ImageNet': 15,
                 "ImageNet_v2": 15, "Gowalla": 7}[dataset] for dataset in
                self.args.dataset]
            self.tw_range = [0.5, 0.1]
            self.models_semi_convergence_flag = [False] * self.ME
            self.models_semi_convergence_count = [0] * self.ME
            self.clients_metrics = {client_id: {"fraction_of_classes": {me: None for me in range(self.ME)},
                                                "imbalance_level": {me: None for me in range(self.ME)},
                                                "train_class_count": {me: None for me in range(self.ME)}} for client_id in
                                    range(1, self.total_clients + 1)}
            self.client_class_count = {me: {i: [] for i in range(self.total_clients)} for me in range(self.ME)}
            self.minimum_training_clients_per_model = {0.1: 1, 0.2: 2, 0.3: 3, 0.5: 3}[self.fraction_fit]
            self.training_clients_per_model_per_round = {me: {t: [] for t in range(1, self.number_of_rounds + 1)} for me in range(self.ME)}
            self.rounds_since_last_semi_convergence = {me: 0 for me in range(self.ME)}
            self.unique_count_samples = {me: np.array([0 for i in range(self.n_classes[me])]) for me in range(self.ME)}
            self.models_semi_convergence_rounds_n_clients = {m: [] for m in range(self.ME)}
            self.accuracy_gain_models = {me: [] for me in range(self.ME)}
            self.stop_cpd = [False for me in range(self.ME)]
            self.re_per_model = int(args.reduction)
            self.fraction_of_classes = np.zeros((self.ME, self.total_clients + 1))
            self.imbalance_level = np.zeros((self.ME, self.total_clients + 1))
            self.lim = []
            self.free_budget_distribution_factor = args.df

            if self.fraction_evaluate != 1.0:
                raise ValueError("fraction_evaluate must be 1.0 to run MultiFedEfficiency")
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
            sample_size, min_num_clients = self.num_fit_clients(
                client_manager.num_available()
            )

            n_clients = int(self.total_clients * self.fraction_fit)

            logger.info("""Rodada {} clients metrics {}""".format(server_round, self.clients_metrics))

            logging.info("""sample clientes {} {} disponiveis {} rodada {} n clients {}""".format(sample_size, min_num_clients, client_manager.num_available(), server_round, n_clients))
            clients = client_manager.sample(
                num_clients=n_clients, min_num_clients=n_clients
            )

            # if server_round >= 3:
            selected_clients_m = self.random_selection(server_round, clients)
            logger.info("""selecionados random {}""".format([len(i) for i in selected_clients_m]))

            n = len(clients) // self.ME
            # selected_clients_m = np.array_split(clients, self.ME)

            self.n_trained_clients = len(clients)
            logging.info("""selecionados {} por modelo {} rodada {}""".format(self.n_trained_clients, [len(i) for i in selected_clients_m], server_round))

            # Return client/config pairs
            clients_m = []
            for me in range(self.ME):
                sc = selected_clients_m[me]
                for client in sc:
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

    def aggregate_evaluate(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, EvaluateRes]],
        failures: list[Union[tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> tuple[Optional[float], dict[str, Scalar]]:
        """Aggregate evaluation losses using weighted average."""
        try:
            if not results:
                return None, {}
            # Do not aggregate if there are failures and failures are not accepted
            if not self.accept_failures and failures:
                return None, {}

            logger.info("""inicio aggregate evaluate {} quantidade de clientes recebidos {}""".format(server_round, len(results)))

            for i in range(len(results)):
                _, result = results[i]
                for me in result.metrics:
                    tuple_me = pickle.loads(result.metrics[str(me)])
                    results_mefl = tuple_me[2]
                    client_id = results_mefl["client_id"]
                    self.training_clients_per_model_per_round[int(me)][server_round].append(client_id)

            if server_round == 2 and self.fraction_evaluate == 1.0:
                for i in range(len(results)):
                    _, result = results[i]
                    for me in result.metrics:
                        tuple_me = pickle.loads(result.metrics[str(me)])
                        results_mefl = tuple_me[2]
                        fraction_of_classes = results_mefl["fraction_of_classes"]
                        imbalance_level = results_mefl["imbalance_level"]
                        train_class_count = results_mefl["train_class_count"]
                        client_id = results_mefl["client_id"]
                        self.clients_metrics[client_id]["fraction_of_classes"][int(me)] = fraction_of_classes
                        self.clients_metrics[client_id]["imbalance_level"][int(me)] = imbalance_level
                        self.clients_metrics[client_id]["train_class_count"][int(me)] = train_class_count

            if server_round >= 2:
                if server_round == 2:
                    self.calculate_non_iid_degree_of_models()
                self.process(server_round)
            # if server_round == 12:
            # #     logger.info("""Média fraction of classes: {}""".format(np.mean(self.fraction_of_classes, axis=1)))
            # #     logger.info("""Média imbalance level: {}""".format(np.mean(self.imbalance_level, axis=1)))
            # #     logger.info("""Need for training: {}""".format(self.need_for_training))
            #     exit()

            return super().aggregate_evaluate(server_round, results, failures)
        except Exception as e:
            logger.error("aggregate_evaluate error")
            logger.error("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))

    def calculate_non_iid_degree_of_models(self):
        try:
            logger.info(f"entrou no calculate non iid")

            for me in range(self.ME):
                for i in self.clients_metrics.keys():
                    # non-iid degree
                    self.fraction_of_classes[me][i] = self.clients_metrics[i]["fraction_of_classes"][me]
                    self.imbalance_level[me][i] = self.clients_metrics[i]["imbalance_level"][me]

            logger.info(self.dataset)
            average_fraction_of_classes = 1 - np.mean(self.fraction_of_classes, axis=1)
            average_balance_level = np.mean(self.imbalance_level, axis=1)
            self.need_for_training = (average_fraction_of_classes + average_balance_level) / 2
            weighted_need_for_training = self.need_for_training / np.sum(self.need_for_training)

            logger.info("""Média fraction of classes: {}""".format(np.mean(self.fraction_of_classes, axis=1)))
            logger.info("""Média imbalance level: {}""".format(np.mean(self.imbalance_level, axis=1)))
            logger.info("""Need for training: {}""".format(self.need_for_training))
            logger.info("""Weighted need for training: {}""".format(weighted_need_for_training))
        except Exception as e:
            logger.error("calculate_non_iid_degree_of_models error")
            logger.error("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))

    def process(self, t: int):
        """semi-convergence detection"""
        try:
            if t == 2:
                diff = self.tw_range[0] - self.tw_range[1]
                middle = self.tw_range[1] + diff // 2
                for me in range(self.ME):
                    # method 2
                    # r = diff * (1 - self.need_for_training[m])
                    # if self.need_for_training[m] >= 0.5:
                    #     # Makes it harder to reduce training intensity
                    #     lower = upper - t/2
                    #     upper = lower + diff * (1- self.need_for_training[m])
                    # else:
                    #     # Makes it easier to reduce training intensity
                    #     upper = self.tw_range[0]
                    #     lower = upper - diff * self.need_for_training[m]

                    # method 1 funciona bem
                    lower = self.tw_range[1]
                    upper = self.tw_range[0]
                    r = diff * (1 - self.need_for_training[me])
                    # Smaller training reduction interval for higher need for training
                    lower = max(middle - r / 2, lower)
                    upper = min(middle + r / 2, upper)

                    self.lim.append([upper, lower])
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
                    losses = self.results_test_metrics[me]["Loss"][-(self.tw[me] + 1):]
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

            budget = int((self.total_clients * self.fraction_fit) / self.ME)
            cm = [budget] * self.ME

            for me in range(self.ME):
                if self.models_semi_convergence_count[me] > 0:
                    cm[me] = int(max(self.minimum_training_clients_per_model,
                                                      cm[me] - self.models_semi_convergence_count[me]))

            logger.info("""cm i: {}""".format(cm))

            if self.free_budget_distribution_factor > 0 and t >= 3:
                free_budget = int((self.total_clients * self.fraction_fit) - np.sum(cm))
                k_nt = len(np.argwhere(self.need_for_training >= 0.34))
                if free_budget > 0 and k_nt > 0:
                    free_budget_k = int(int(free_budget * self.free_budget_distribution_factor) / k_nt)
                    rest = free_budget - int(free_budget_k * k_nt)

                    logger.info(f"Free budget: {free_budget},  k nt: {k_nt}, Free budget k: {free_budget_k}, resto: {rest}")

                    for me in range(self.ME):
                        if self.need_for_training[me] >= 0.5 and cm[me] == budget:
                            cm[me] = int(cm[me] + free_budget_k)
                            if rest > 0:
                                cm[me] += 1
                                rest -= 1
                                rest = max(rest, 0)

            selected_clients = np.random.choice(clients, int(self.fraction_fit * self.total_clients), replace=False).tolist()
            selected_clients = [i for i in selected_clients]

            selected_clients_m = [None] * self.ME

            logger.info("""a : {}""".format(selected_clients_m))
            logger.info("""random: {}""".format(selected_clients))
            logger.info("""cm: {}""".format(cm))
            i = 0
            reverse_list = [k for k in range(self.ME)]
            for me in reverse_list:
                j = i + cm[me]
                selected_clients_m[me] = selected_clients[i: j]
                i = j

            return selected_clients_m

        except Exception as e:
            logger.error("Error random selection")
            logger.error('Error on line {} {} {}'.format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))