import logging

import csv
import os

import torch
import numpy as np
from typing import List, Tuple
from flwr.common import Metrics

from sklearn.utils.extmath import randomized_svd
from typing import Callable, Optional, Union

from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
)
from flwr.common.logger import log
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.aggregate import aggregate, aggregate_inplace, weighted_loss_avg

import sys
import flwr

from logging import WARNING
import random
from itertools import islice

from fedpredict.utils.compression_methods.parameters_svd import parameter_svd_write, inverse_parameter_svd_reading
from fedpredict.fedpredict_core import layer_compression_range
from fedpredict.utils.compression_methods.fedkd import fedkd_compression
from utils.models_utils import load_model, get_weights_fedkd, load_data, set_weights, test, train
from logging import WARNING
from flwr.common import FitIns
from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg
from flwr.common.logger import log

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

from server.FL.server_fedavg import FedAvg

def aggregate(results: list[tuple[NDArrays, int]], original_parameters: list) -> NDArrays:
    """Compute weighted average."""
    # Calculate the total number of examples used during training
    num_examples_total = sum(num_examples for (_, num_examples) in results)

    # Create a list of weights, each multiplied by the related number of examples
    weighted_weights = [
        [(original_layer + layer) * num_examples for layer, original_layer in zip(weights, original_parameters)] for weights, num_examples in results
    ]

    # Compute average weights of each layer
    weights_prime: NDArrays = [
        reduce(np.add, layer_updates) / num_examples_total
        for layer_updates in zip(*weighted_weights)
    ]
    return weights_prime

# Initialize Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define metric aggregation function
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["Accuracy"] for num_examples, m in metrics]
    balanced_accuracies = [num_examples * m["Balanced accuracy"] for num_examples, m in metrics]
    loss = [num_examples * m["Loss"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"Accuracy": sum(accuracies) / sum(examples), "Balanced accuracy": sum(balanced_accuracies) / sum(examples),
            "Loss": sum(loss) / sum(examples), "Round (t)": metrics[0][1]["Round (t)"], "Model size": metrics[0][1]["Model size"]}

def weighted_loss_avg(results: list[tuple[int, float]]) -> float:
    """Aggregate evaluation results obtained from multiple clients."""
    num_total_evaluation_examples = sum(num_examples for (num_examples, _) in results)
    weighted_losses = [num_examples * loss for num_examples, loss in results]
    return sum(weighted_losses) / num_total_evaluation_examples

def if_reduces_size(shape, n_components, dtype=np.float64):

    try:
        size = np.array([1], dtype=dtype)
        p = shape[0]
        q = shape[1]
        k = n_components

        if p*k + k*k + k*q < p*q:
            return True
        else:
            return False

    except Exception as e:
        logger.info("svd")
        logger.info('Error on line {} {} {}'.format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))

def layer_compression_range(model_shape):

    try:
        layers_range = []
        for shape in model_shape:

            layer_range = 0
            if len(shape) >= 2:
                shape = shape[-2:]

                col = shape[1]
                for n_components in range(1, col+1):
                    if if_reduces_size(shape, n_components):
                        layer_range = n_components
                    else:
                        break

            layers_range.append(layer_range)

        return layers_range

    except Exception as e:
        logger.info("layer_compression_range")
        logger.info('Error on line {} {} {}'.format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))

def svd(layer, n_components, svd_type='tsvd'):

    try:
        np.random.seed(0)
        # print("ola: ", int(len(layer) * n_components), layer.shape, layer)
        if n_components > 0 and n_components < 1:
            n_components = int(len(layer) * n_components)

        if svd_type == 'tsvd':
            U, Sigma, VT = randomized_svd(layer,
                                          n_components=n_components,
                                          n_iter=5,
                                          random_state=0)
        else:
            U, Sigma, VT = np.linalg.svd(layer, full_matrices=False)
            U = U[:, :n_components]
            Sigma = Sigma[:n_components]
            VT = VT[:n_components, :]

        # print(U.shape, Sigma.shape, VT.T.shape)
        return [U, VT, Sigma]

    except Exception as e:
        logger.info("svd")
        logger.info('Error on line {} {} {}'.format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))

def parameter_svd(layer, n_components, svd_type='tsvd'):

    try:
        if np.ndim(layer) == 1 or n_components is None:
            return [layer, np.array([]), np.array([])]
        elif np.ndim(layer) == 2:
            r = svd(layer, n_components, svd_type)
            return r
        elif np.ndim(layer) >= 3:
            u = []
            v = []
            sig = []
            for i in range(len(layer)):
                r = parameter_svd(layer[i], n_components, svd_type)
                u.append(r[0])
                v.append(r[1])
                sig.append(r[2])
            return [np.array(u), np.array(v), np.array(sig)]

    except Exception as e:
        logger.info("parameter_svd")
        logger.info('Error on line {} {} {}'.format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))

def parameter_svd_write(arrays, n_components_list, svd_type='tsvd'):

    try:

        u = []
        vt = []
        sigma_parameters = []
        arrays_compre = []
        for i in range(len(arrays)):
            if type(n_components_list) == list:
                n_components = n_components_list[i]
            else:
                n_components = n_components_list
            # logger.info("Indice da camada: ", i)
            r = parameter_svd(arrays[i], n_components, svd_type)
            arrays_compre += r

        return arrays_compre

    except Exception as e:
        logger.info("paramete_svd")
        logger.info('Error on line {} {} {}'.format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))

# pylint: disable=line-too-long
class FedKD(FedAvg):
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
        Initial global utils parameters.
    fit_metrics_aggregation_fn : Optional[MetricsAggregationFn]
        Metrics aggregation function, optional.
    evaluate_metrics_aggregation_fn : Optional[MetricsAggregationFn]
        Metrics aggregation function, optional.
    inplace : bool (default: True)
        Enable (True) or disable (False) in-place aggregation of utils updates.
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
            super().__init__(args=args, fraction_fit=fraction_fit, fraction_evaluate=fraction_evaluate,
                             min_fit_clients=min_fit_clients, min_evaluate_clients=min_evaluate_clients,
                             min_available_clients=min_available_clients, evaluate_fn=evaluate_fn,
                             on_fit_config_fn=on_fit_config_fn, on_evaluate_config_fn=on_evaluate_config_fn,
                             accept_failures=accept_failures, initial_parameters=initial_parameters,
                             fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
                             evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn, inplace=inplace)
            self.model = get_weights_fedkd(load_model(args.model[0], args.dataset[0], args.strategy, args.device))
            self.model_shape = [i.shape for i in self.model]
            self.layers_compression_range = layer_compression_range(self.model_shape)
        except Exception as e:
            logger.error("__init__ error")
            logger.error("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> list[tuple[ClientProxy, FitIns]]:
        try:
            torch.random.manual_seed(server_round)
            random.seed(server_round)
            np.random.seed(server_round)
            client_manager.wait_for(self.total_clients, 1000)
            """Configure the next round of training."""
            config = {}
            if self.on_fit_config_fn is not None:
                # Custom fit config function provided
                config = self.on_fit_config_fn(server_round)
            n_components_list = []
            initial_parameters = parameters_to_ndarrays(parameters)
            self.original_parameters = parameters_to_ndarrays(parameters)
            if server_round > 1:
                for i in range(len(initial_parameters)):
                    compression_range = self.layers_compression_range[i]
                    if compression_range > 0:
                        compression_range = self.fedkd_formula(server_round, self.number_of_rounds, compression_range)
                    else:
                        compression_range = None
                    n_components_list.append(compression_range)

                logger.info(f"n_components_list: {n_components_list}")
                # parameters = ndarrays_to_parameters(
                #     parameter_svd_write(initial_parameters, n_components_list, 'svd'))
                parameters, layers_fraction = fedkd_compression(0, self.layers_compression_range, self.number_of_rounds, server_round, len(self.layers_compression_range),
                                                                        parameters_to_ndarrays(parameters))
                parameters = self.compress(server_round, parameters_to_ndarrays(parameters))

            config["t"] = server_round
            fit_ins = FitIns(parameters, config)

            if server_round == 1:
                self.model_shape = [i.shape for i in flwr.common.parameters_to_ndarrays(parameters)]

            # Insert new clients
            if server_round < self.experiment_config["round_of_new_clients"]:
                self.current_total_clients = self.experiment_config["initial_number_of_clients"]
                # Limit the number of available clients
                clients = dict(islice(client_manager.clients.items(), self.current_total_clients))
                clients = [clients[key] for key in clients.keys()]
            else:
                self.current_total_clients = len(client_manager.clients)
                clients = client_manager.clients
                clients = [clients[key] for key in clients.keys()]

            # Sample clients
            sample_size, min_num_clients = self.num_fit_clients(
                client_manager.num_available()
            )


            n_clients = int(self.total_clients * self.fraction_fit)

            logging.info("""sample clientes {} {} disponiveis {} rodada {} n clients {}""".format(sample_size, min_num_clients, client_manager.num_available(), server_round, n_clients))
            logger.info(f"available clients {len(clients)} round {server_round} av {type(clients)}")
            logger.info(f"population {len(clients)} samples {n_clients} round {server_round} confi")
            clients = np.random.choice(clients, size=min([n_clients, len(clients)]), replace=False)

            self.n_trained_clients = len(clients)
            self.selected_clients = [client.cid for client in clients]
            self.selected_clients_ids = 0
            logging.info("""selecionados {} rodada {}""".format(self.n_trained_clients, server_round))

            # Return client/config pairs
            return [(client, fit_ins) for client in clients]
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

            if self.inplace:
                # Does in-place weighted average of results
                aggregated_ndarrays = aggregate_inplace(results)
            else:
                # Convert results
                weights_results = [
                    (inverse_parameter_svd_reading(parameters_to_ndarrays(fit_res.parameters)), fit_res.num_examples)
                    for _, fit_res in results
                ]
                aggregated_ndarrays = aggregate(weights_results, self.original_parameters)

            parameters_aggregated = ndarrays_to_parameters(aggregated_ndarrays)

            # Aggregate custom metrics if aggregation fn was provided
            metrics_aggregated = {}
            if self.fit_metrics_aggregation_fn:
                fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
                metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
            elif server_round == 1:  # Only log this warning once
                log(WARNING, "No fit_metrics_aggregation_fn provided")

            return parameters_aggregated, metrics_aggregated

        except Exception as e:
            logger.error("aggregate_fit error")
            logger.error("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> list[tuple[ClientProxy, EvaluateIns]]:
        try:
            """Configure the next round of evaluation."""
            # Do not configure federated evaluation if fraction eval is 0.
            logger.info(f"Início configure evaluate {server_round}")
            if self.fraction_evaluate == 0.0:
                return []

            # Parameters and config
            config = {}
            if self.on_evaluate_config_fn is not None:
                # Custom evaluation config function provided
                config = self.on_evaluate_config_fn(server_round)

            parameters_to_send = parameters_to_ndarrays(parameters)
            n_components_list = []
            if server_round > 1:
                for i in range(len(parameters_to_send)):
                    compression_range = self.layers_compression_range[i]
                    if compression_range > 0:
                        compression_range = self.fedkd_formula(server_round, self.number_of_rounds, compression_range)
                    else:
                        compression_range = None
                    n_components_list.append(compression_range)

                parameters_to_send = ndarrays_to_parameters(
                    parameter_svd_write(parameters_to_send, n_components_list, 'svd'))
                parameters = parameters_to_send

            config["t"] = server_round
            evaluate_ins = EvaluateIns(parameters, config)

            # Insert new clients
            if server_round < self.experiment_config["round_of_new_clients"]:
                self.current_total_clients = self.experiment_config["initial_number_of_clients"]
                # Limit the number of available clients
                clients = dict(islice(client_manager.clients.items(), self.current_total_clients))
                clients = [clients[key] for key in clients.keys()]
            else:
                clients = client_manager.clients
                clients = [clients[key] for key in clients.keys()]

            # Sample clients
            sample_size, min_num_clients = self.num_evaluation_clients(
                client_manager.num_available()
            )
            # clients = client_manager.sample(
            #     num_clients=sample_size, min_num_clients=min_num_clients
            # )
            logger.info(f"population {len(clients)} samples {sample_size} round {server_round} eval")
            clients = np.random.choice(clients, size=min([sample_size, len(clients)]), replace=False)

            # exit()

            # Return client/config pairs
            r = [(client, evaluate_ins) for client in clients]
            # for client in r:
            #     logger.info(f"antes type client {type(client)} type 0 {type(client[0])} type 1 {type(client[1])}")
            #     logger.info(f"parameters {type(client[1].parameters)} config {client[1].config}")
            logger.info(f"Fim configure evaluate {server_round}")
            return r
        except Exception as e:
            logger.error("configure_evaluate error")
            logger.error("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))

    def aggregate_evaluate(
            self,
            server_round: int,
            results: list[tuple[ClientProxy, EvaluateRes]],
            failures: list[Union[tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> tuple[Optional[float], dict[str, Scalar]]:
        try:
            """Aggregate evaluation losses using weighted average."""
            logger.info(f"Início aggregate evaluate round {server_round}")
            if not results:
                return None, {}
            # Do not aggregate if there are failures and failures are not accepted
            if not self.accept_failures and failures:
                return None, {}

            # Aggregate loss
            logger.info("""metricas recebidas rodada {}: {}""".format(server_round, results))
            loss_aggregated = weighted_loss_avg(
                [
                    (evaluate_res.num_examples, evaluate_res.loss)
                    for _, evaluate_res in results
                ]
            )

            # Aggregate custom metrics if aggregation fn was provided
            metrics_aggregated = {}
            if self.evaluate_metrics_aggregation_fn:
                eval_metrics = [(res.num_examples, res.metrics) for _, res in results]
                metrics_aggregated = self.evaluate_metrics_aggregation_fn(eval_metrics)
                accuracies = [round(j["Accuracy"], 2) for i, j in eval_metrics]
                nts = [i[1]["nt"] for i in eval_metrics]
            elif server_round == 1:  # Only log this warning once
                log(WARNING, "No evaluate_metrics_aggregation_fn provided")

            if server_round == 1:
                mode = "w"
            else:
                mode = "w"
            data = [[str(accuracies), server_round, str(nts)]]
            self.add_metrics(server_round, metrics_aggregated)
            self.save_results(mode)
            self.save_results_nt(server_round, data)

            logger.info(f"Fim aggregate evaluate round {server_round}")

            return loss_aggregated, metrics_aggregated
        except Exception as e:
            logger.error("aggregate_evaluate error")
            logger.error("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))

    def add_metrics(self, t, metrics_aggregated):
        try:
            metrics_aggregated["Fraction fit"] = self.fraction_fit
            metrics_aggregated["# training clients"] = self.n_trained_clients
            metrics_aggregated["# available clients"] = self.current_total_clients
            metrics_aggregated["training clients and models"] = self.selected_clients_ids
            metrics_aggregated["Alpha"] = self.alpha

            for metric in metrics_aggregated:
                self.results_test_metrics[metric].append(metrics_aggregated[metric])
        except Exception as e:
            logger.error("add_metrics error")
            logger.error("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))

    def save_results(self, mode):
        try:
            # train
            # logger.info("""save results: {}""".format(self.results_test_metrics))
            file_path, header, data = self.get_results( 'train', '')
            # logger.info("""dados: {} {}""".format(data, file_path))
            self._write_header(file_path, header=header, mode=mode)
            self._write_outputs(file_path, data=data)

            # test

            self.file_path, header, data = self.get_results( 'test', '')
            self._write_header(self.file_path, header=header, mode=mode)
            self._write_outputs(self.file_path, data=data)
        except Exception as e:
            logger.error("save_results error")
            logger.error("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))

    def get_results(self, train_test, mode):
        try:
            algo = self.dataset + "_" + self.strategy_name

            # result_path = """/results/concept_drift_{}/new_clients_fraction_{}_round_{}/clients_{}/alpha_{}/alpha_end_{}/{}/concept_drift_rounds_{}_{}/{}/fc_{}/rounds_{}/epochs_{}/{}/""".format(
            #     self.cd,
            #     self.fraction_new_clients,
            #     self.round_new_clients,
            #     self.total_clients,
            #     self.alpha,
            #     self.alpha,
            #     self.dataset,
            #     0,
            #     0,
            #     self.model_name,
            #     self.fraction_fit,
            #     self.number_of_rounds,
            #     self.local_epochs,
            #     train_test)

            result_path = """results/experiment_id_{}/clients_{}/alpha_{}/{}/{}/fc_{}/rounds_{}/epochs_{}/{}/""".format(
                self.experiment_id,
                self.total_clients,
                self.alpha,
                self.dataset,
                self.model_name,
                self.fraction_fit,
                self.number_of_rounds,
                self.local_epochs,
                train_test)

            logger.info(f"caminho {result_path}")


            if not os.path.exists(result_path):
                os.makedirs(result_path)

            file_path = result_path + "{}.csv".format(algo)

            if train_test == 'test':

                header = self.test_metrics_names
                # print(self.rs_test_acc)
                # print(self.rs_test_auc)
                # print(self.rs_train_loss)
                list_of_metrics = []
                for me in self.results_test_metrics:
                    # print(me, len(self.results_test_metrics[me]))
                    length = len(self.results_test_metrics[me])
                    list_of_metrics.append(self.results_test_metrics[me])

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
                    for me in self.results_train_metrics:
                        # print(me, len(self.results_train_metrics[me]))
                        length = len(self.results_train_metrics[me])
                        list_of_metrics.append(self.results_train_metrics[me])

                    data = []
                    logger.info("""tamanho: {}    {}""".format(length, list_of_metrics))
                    for i in range(length):
                        row = []
                        for j in range(len(list_of_metrics)):
                            if len(list_of_metrics[j]) > 0:
                                row.append(list_of_metrics[j][i])
                            else:
                                row.append(0)

                        data.append(row)


            logger.info("File path: " + file_path)
            logger.info(data)

            return file_path, header, data
        except Exception as e:
            logger.error("get-results error")
            logger.error("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))

    def _write_header(self, filename, header, mode):

        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, mode) as server_log_file:
            writer = csv.writer(server_log_file)
            writer.writerow(header)

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

    def set_experiment_config(self, experiment_id):
        try:
            logger.info(f"id do experimento {experiment_id}")
            if "new_clients" in experiment_id:
                round_of_new_clients = int(self.number_of_rounds * 0.7)
                initial_number_of_clients = int(self.total_clients * 0.7)
                return {"round_of_new_clients": round_of_new_clients, "initial_number_of_clients": initial_number_of_clients}

            else:
                return {"round_of_new_clients": 0, "initial_number_of_clients": self.total_clients}

        except Exception as e:
            logger.error("set_experiment_config error")
            logger.error("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))

    def save_results_nt(self, server_round, data):
        try:
            algo = self.dataset + "_" + self.strategy_name
            result_path = """results/experiment_id_{}/clients_{}/alpha_{}/{}/{}/fc_{}/rounds_{}/epochs_{}/{}/""".format(
                self.experiment_id,
                self.total_clients,
                self.alpha,
                self.dataset,
                self.model_name,
                self.fraction_fit,
                self.number_of_rounds,
                self.local_epochs,
                "test")
            compression = ""
            if len(compression) > 0:
                compression = "_" + compression
            result_path = "{}{}{}_nt.csv".format(result_path, algo, compression)
            if server_round == 1:
                self._write_header(result_path, header=self.test_metrics_names_nt, mode="w")
            self._write_outputs(result_path, data=data)
        except Exception as e:
            logger.error("save_results error")
            logger.error("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))

    def fedkd_formula(self, server_round, num_rounds, compression_range):

        frac = max(1, abs(1 - server_round)) / num_rounds
        compression_range = max(round(frac * compression_range), 1)
        logger.info(f"compression range: {compression_range} rounds: {server_round}")
        return compression_range

    def compress(self, server_round, parameters):

        try:
            layers_compression_range = self.layers_compression_range([i.shape for i in parameters])
            n_components_list = []
            for i in range(len(parameters)):
                compression_range = layers_compression_range[i]
                if compression_range > 0:
                    frac = 1 - server_round / self.number_of_rounds
                    compression_range = max(round(frac * compression_range), 1)
                else:
                    compression_range = None
                n_components_list.append(compression_range)

            parameters_to_send = parameter_svd_write(parameters, n_components_list)
            return [Parameter(torch.Tensor(i.tolist())) for i in parameters_to_send]

        except Exception as e:
            logger.info("compress")
            logger.info('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)