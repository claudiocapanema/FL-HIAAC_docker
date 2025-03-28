import logging

import csv
import os
import sys
import json
import pickle

import torch
import numpy as np
from typing import List, Tuple
from flwr.common import Metrics

from typing import Callable, Optional, Union

from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.common.logger import log
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.aggregate import aggregate, aggregate_inplace, weighted_loss_avg


import flwr

from logging import WARNING
import random

# Initialize Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def weighted_loss_avg(results: list[tuple[int, float]]) -> float:
    """Aggregate evaluation results obtained from multiple clients."""
    num_total_evaluation_examples = sum(num_examples for (num_examples, _) in results)
    weighted_losses = [num_examples * loss for num_examples, loss in results]
    return sum(weighted_losses) / num_total_evaluation_examples


def global_concept_dirft_config(ME, n_rounds, alphas, experiment_id, seed=0):
    # np.random.seed(seed)
    # fraction_min_round = 0.3
    # min_round = int(fraction_min_round * n_rounds)
    #
    # new_alphas = np.random.choice(alphas, ME, replace=True)
    # while new_alphas == alphas:
    #     np.random.seed()
    #     new_alphas = np.random.choice(alphas, ME, replace=True)

    np.random.seed(seed)
    if experiment_id > 0:
        if experiment_id == 1:
            ME_concept_drift_rounds = [[int(n_rounds * 0.3)], [int(n_rounds) * 0.6]]
            new_alphas = [[alphas[1]], [alphas[0]]]

        config = {me: {"concept_drift_rounds": ME_concept_drift_rounds[me], "new_alphas": ME_concept_drift_rounds[me]} for me in range(len(ME_concept_drift_rounds))}
    else:
        config = {}
    return config


# pylint: disable=line-too-long
class MultiFedAvg(flwr.server.strategy.FedAvg):
    """Federated Averaging strategy.

    Implementation based on https://arxiv.org/abs/1602.05629

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
                             evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn, inplace=inplace)
            self.args = args
            self.local_epochs = args.local_epochs
            self.fraction_new_clients = args.fraction_new_clients
            self.round_new_clients = args.round_new_clients
            self.alpha = [float(i) for i in args.alpha]
            self.total_clients = args.total_clients
            self.dataset = args.dataset
            self.model_name = args.model
            self.ME = len(self.model_name)
            self.number_of_rounds = args.number_of_rounds
            self.cd = "false" if args.concept_drift_experiment_id == 0 else f"true_experiment_id_{args.concept_drift_experiment_id}"
            self.strategy_name = args.strategy
            self.test_metrics_names = ["Accuracy", "Balanced accuracy", "Loss", "Round (t)", "Fraction fit",
                                       "# training clients", "training clients and models", "Model size", "Alpha"]
            self.train_metrics_names = ["Accuracy", "Balanced accuracy", "Loss", "Round (t)", "Fraction fit",
                                       "# training clients", "training clients and models", "Model size", "Alpha"]
            self.rs_test_acc = {me: [] for me in range(self.ME)}
            self.rs_test_auc = {me: [] for me in range(self.ME)}
            self.rs_train_loss = {me : [] for me in range(self.ME)}
            self.results_train_metrics = {me: {metric: [] for metric in self.train_metrics_names} for me in range(self.ME)}
            self.results_train_metrics_w = {me: {metric: [] for metric in self.train_metrics_names} for me in range(self.ME)}
            self.results_test_metrics = {me: {metric: [] for metric in self.test_metrics_names} for me in range(self.ME)}
            self.results_test_metrics_w = {me: {metric: [] for metric in self.test_metrics_names} for me in range(self.ME)}
            self.clients_results_test_metrics = {me: {metric: [] for metric in self.test_metrics_names} for me in range(self.ME)}
            self.selected_clients_m = []
            self.selected_clients_m_ids_random = [[] for me in range(self.ME)]
            # Concept drift parameters
            self.concept_drift_experiment_id = args.concept_drift_experiment_id
            self.concept_drift_config = global_concept_dirft_config(self.ME, self.number_of_rounds, self.alpha, self.concept_drift_experiment_id, 0)
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

            logging.info("""sample clientes {} {} disponiveis {} rodada {} n clients {}""".format(sample_size, min_num_clients, client_manager.num_available(), server_round, n_clients))
            clients = client_manager.sample(
                num_clients=n_clients, min_num_clients=n_clients
            )

            n = len(clients) // self.ME
            selected_clients_m = np.array_split(clients, self.ME)

            self.n_trained_clients = len(clients)
            logging.info("""selecionados {} por modelo {} rodada {}""".format(self.n_trained_clients, [len(i) for i in selected_clients_m], server_round))

            # Return client/config pairs
            clients_m = []
            for me in range(self.ME):
                sc = selected_clients_m[me]
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

            results_mefl = {me: [] for me in range(self.ME)}
            for i in range(len(results)):
                _, result = results[i]
                me = result.metrics["me"]
                client_id = result.metrics["client_id"]
                self.selected_clients_m[me].append(client_id)
                results_mefl[me].append(results[i])


            aggregated_ndarrays_mefl = {me: None for me in range(self.ME)}
            aggregated_ndarrays_mefl = {me: [] for me in range(self.ME)}
            weights_results_mefl = {me: [] for me in range(self.ME)}
            parameters_aggregated_mefl = {me: [] for me in range(self.ME)}

            if self.inplace:
                for me in range(self.ME):
                    # Does in-place weighted average of results
                    aggregated_ndarrays_mefl[me] = aggregate_inplace(results_mefl[me])
            else:
                for me in range(self.ME):
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

            for me in range(self.ME):
                logger.info(f"tamanho para modelo {me} rodada {server_round} {len(aggregated_ndarrays_mefl[me])}")
                parameters_aggregated_mefl[me] = ndarrays_to_parameters(aggregated_ndarrays_mefl[me])

            # Aggregate custom metrics if aggregation fn was provided
            metrics_aggregated_mefl = {me: [] for me in range(self.ME)}
            for me in range(self.ME):
                if self.fit_metrics_aggregation_fn:
                    fit_metrics = [(res.num_examples, res.metrics) for _, res in results_mefl[me]]
                    metrics_aggregated_mefl[me] = self.fit_metrics_aggregation_fn(fit_metrics)
                elif server_round == 1:  # Only log this warning once
                    log(WARNING, "No fit_metrics_aggregation_fn provided")

            logger.info("""finalizou aggregated fit""")
            logger.info(f"finalizou aggregated fit {server_round} {metrics_aggregated_mefl}")

            self.parameters_aggregated_mefl = parameters_aggregated_mefl
            self.metrics_aggregated_mefl = metrics_aggregated_mefl

            return parameters_aggregated_mefl, metrics_aggregated_mefl
        except Exception as e:
            logger.error("aggregate_fit error")
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

            logger.info("""inicio aggregate evaluate {}""".format(server_round))

            results_mefl = {me: [] for me in range(self.ME)}
            for i in range(len(results)):
                _, result = results[i]
                for me in result.metrics:
                    results_mefl[int(me)].append(pickle.loads(result.metrics[str(me)]))


            # Aggregate loss
            # logging.info("""metricas recebidas rodada {}: {}""".format(server_round, results_mefl[0]))
            loss_aggregated_mefl = {me: 0. for me in range(self.ME)}
            for me in results_mefl.keys():
                loss_aggregated = weighted_loss_avg(
                    [
                        (num_examples, loss)
                        for loss, num_examples, metrics in results_mefl[me]
                    ]
                )
                loss_aggregated_mefl[int(me)] = loss_aggregated

            # Aggregate custom metrics if aggregation fn was provided
            metrics_aggregated_mefl = {me: {} for me in range(self.ME)}
            if self.evaluate_metrics_aggregation_fn:
                for me in results_mefl.keys():
                    eval_metrics = [(num_examples, metrics) for loss, num_examples, metrics in results_mefl[me]]
                    metrics_aggregated_mefl[int(me)] = self.evaluate_metrics_aggregation_fn(eval_metrics)
            elif server_round == 1:  # Only log this warning once
                log(WARNING, "No evaluate_metrics_aggregation_fn provided")

            mode = "w"

            for me in range(self.ME):
                self.add_metrics(server_round, metrics_aggregated_mefl, me)
                self._save_results(mode, me)


            return loss_aggregated_mefl, metrics_aggregated_mefl
        except Exception as e:
            logger.error("aggregate_evaluate error")
            logger.error("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))

    def add_metrics(self, server_round, metrics_aggregated, me):
        try:
            metrics_aggregated[me]["Fraction fit"] = self.fraction_fit
            metrics_aggregated[me]["# training clients"] = self.n_trained_clients
            metrics_aggregated[me]["training clients and models"] = self.selected_clients_m[me]

            for metric in metrics_aggregated[me]:
                self.results_test_metrics[me][metric].append(metrics_aggregated[me][metric])
        except Exception as e:
            logger.error("add_metrics error")
            logger.error("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))

    def _save_results(self, mode, me):

        # train
        try:
            # logger.info("""save results: {}""".format(self.results_test_metrics[me]))
            file_path, header, data = self._get_results('train', '', me)
            # logger.info("""dados: {} {}""".format(data, file_path))
            self._write_header(file_path, header=header, mode=mode)
            self._write_outputs(file_path, data=data)

            # test

            file_path, header, data = self._get_results('test', '', me)
            self._write_header(file_path, header=header, mode=mode)
            self._write_outputs(file_path, data=data)
        except Exception as e:
            logger.error("save_results error")
            logger.error("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))

    def _get_results(self, train_test, mode, me):

        try:
            algo = self.dataset[me] + "_" + self.strategy_name

            result_path = self.get_result_path(train_test)


            if not os.path.exists(result_path):
                os.makedirs(result_path)

            file_path = result_path + "{}.csv".format(algo)

            if train_test == 'test':

                header = self.test_metrics_names
                # print(self.rs_test_acc[me])
                # print(self.rs_test_auc[me])
                # print(self.rs_train_loss[me])
                list_of_metrics = []
                for metric in self.results_test_metrics[me]:
                    # print(me, len(self.results_test_metrics[me][metric]))
                    length = len(self.results_test_metrics[me][metric])
                    list_of_metrics.append(self.results_test_metrics[me][metric])

                data = []
                for i in range(length):
                    row = []
                    for j in range(len(list_of_metrics)):
                        row.append(list_of_metrics[j][i])

                    data.append(row)

            else:
                if mode == '':
                    header = self.train_metrics_names
                    list_of_metrics = []
                    for metric in self.results_train_metrics[me]:
                        # print(me, len(self.results_train_metrics[me][metric]))
                        length = len(self.results_train_metrics[me][metric])
                        list_of_metrics.append(self.results_train_metrics[me][metric])

                    data = []
                    # logger.info("""tamanho: {}    {}""".format(length, list_of_metrics))
                    for i in range(length):
                        row = []
                        for j in range(len(list_of_metrics)):
                            if len(list_of_metrics[j]) > 0:
                                row.append(list_of_metrics[j][i])
                            else:
                                row.append(0)

                        data.append(row)


            # logger.info("File path: " + file_path)
            logger.info(data)

            return file_path, header, data
        except Exception as e:
            logger.error("get_results error")
            logger.error("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))

    def _write_header(self, filename, header, mode):
        try:
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            with open(filename, mode) as server_log_file:
                writer = csv.writer(server_log_file)
                writer.writerow(header)
        except Exception as e:
            logger.error("_write_header error")
            logger.error("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))

    def _write_outputs(self, filename, data, mode='a'):
        try:
            for i in range(len(data)):
                for j in range(len(data[i])):
                    element = data[i][j]
                    if type(element) == float:
                        element = round(element, 6)
                        data[i][j] = element
            with open(filename, 'a') as server_log_file:
                writer = csv.writer(server_log_file)
                writer.writerows(data)
        except Exception as e:
            logger.error("_write_outputs error")
            logger.error("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))

    def _get_models_size(self):
        try:
            models_size = []
            for me in range(self.ME):
                model = self.global_model[me]
                parameters = [i.detach().cpu().numpy() for i in model.parameters()]
                size = 0
                for i in range(len(parameters)):
                    size += parameters[i].nbytes
                models_size.append(size)
            self.models_size = models_size
        except Exception as e:
            logger.error("_get_models_size error")
            logger.error("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))

    def get_result_path(self, train_test):

        result_path = """/results/concept_drift_{}/new_clients_fraction_{}_round_{}/clients_{}/alpha_{}/{}/{}/fc_{}/rounds_{}/epochs_{}/{}/""".format(
            self.cd,
            self.fraction_new_clients,
            self.round_new_clients,
            self.total_clients,
            self.alpha,
            self.dataset,
            self.model_name,
            self.fraction_fit,
            self.number_of_rounds,
            self.local_epochs,
            train_test)

        return result_path

    def write_log(self, result_path):

        # Abra um arquivo para gravação
        result_path = """{}/log_{}""".format(result_path, self.strategy_name)
        os.makedirs(os.path.dirname(result_path), exist_ok=True)
        with open(result_path, 'w') as f:
            # Redirecione a saída padrão para o arquivo
            original = sys.stdout
            sys.stdout = f
            sys.stdout = original
