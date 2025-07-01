from flwr.common import (
    EvaluateIns,
    parameters_to_ndarrays
)
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from itertools import islice
from fedpredict import fedpredict_server, fedpredict_layerwise_similarity

from logging import WARNING
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

from flwr.server.strategy.aggregate import aggregate, aggregate_inplace



import logging
import numpy as np

from typing import Callable, Optional

from flwr.common import (
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar
)
from server.FL.server_fedavg import FedAvg
import os
import sys
import pickle

# Initialize Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# pylint: disable=line-too-long
class FedAvgFP(FedAvg):
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
            super().__init__(args=args, fraction_fit=fraction_fit, fraction_evaluate=fraction_evaluate, min_fit_clients=min_fit_clients, min_evaluate_clients=min_evaluate_clients, min_available_clients=min_available_clients, evaluate_fn=evaluate_fn, on_fit_config_fn=on_fit_config_fn, on_evaluate_config_fn=on_evaluate_config_fn, accept_failures=accept_failures, initial_parameters=initial_parameters, fit_metrics_aggregation_fn=fit_metrics_aggregation_fn, evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn, inplace=inplace)
            self.compression = "dls_compredict"
            self.similarity_list_per_layer = {}
            self.initial_similarity = 0
            self.current_similarity = 0
            self.model_shape_mefl = None
            self.similarity_between_layers_per_round = {}
            self.similarity_between_layers_per_round_and_client = {}
            self.mean_similarity_per_round = {}
            self.df = 0
            #self.file_path = None
        except Exception as e:
            logger.error("__init__ error")
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

            clients_parameters = []
            for i in range(len(results)):
                _, result = results[i]
                client_id = self.selected_clients[i]
                lt = result.metrics["lt"]
                self.clients_lt[client_id] = lt
                clients_parameters.append(parameters_to_ndarrays(result.parameters))

            if self.inplace:
                # Does in-place weighted average of results
                aggregated_ndarrays = aggregate_inplace(results)
            else:
                # Convert results
                weights_results = [
                    (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
                    for _, fit_res in results
                ]
                aggregated_ndarrays = aggregate(weights_results)

            parameters_aggregated = ndarrays_to_parameters(aggregated_ndarrays)

            # Aggregate custom metrics if aggregation fn was provided
            metrics_aggregated = {}
            if self.fit_metrics_aggregation_fn:
                fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
                metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
            elif server_round == 1:  # Only log this warning once
                log(WARNING, "No fit_metrics_aggregation_fn provided")

            np.random.seed(server_round)
            flag = bool(int(np.random.binomial(1, 0.2, 1)))
            if server_round == 1:
                flag = True

            if "dls" in self.compression:
                if flag:
                    # logger.info(f"tipo: {[type(i) for i in parameters_to_ndarrays(parameters_aggregated)]}")
                    # exit()
                    self.similarity_between_layers_per_round_and_client[server_round], \
                        self.similarity_between_layers_per_round[server_round], self.mean_similarity_per_round[
                        server_round], self.similarity_list_per_layer = fedpredict_layerwise_similarity(
                        global_parameter=parameters_to_ndarrays(parameters_aggregated), clients_parameters=clients_parameters,
                        similarity_per_layer_list=self.similarity_list_per_layer)
                    self.df = float(max(0, abs(np.mean(self.similarity_list_per_layer[0]) - np.mean(
                        self.similarity_list_per_layer[len(parameters_to_ndarrays(parameters_aggregated)) - 2]))))
                else:
                    self.similarity_between_layers_per_round_and_client[server_round], \
                    self.similarity_between_layers_per_round[
                        server_round], self.mean_similarity_per_round[
                        server_round], self.similarity_list_per_layer = self.similarity_between_layers_per_round_and_client[
                        server_round - 1], self.similarity_between_layers_per_round[
                        server_round - 1], self.mean_similarity_per_round[
                        server_round - 1], self.similarity_list_per_layer
            else:
                self.similarity_between_layers_per_round[server_round] = []
                self.mean_similarity_per_round[server_round] = 0
                self.similarity_between_layers_per_round_and_client[server_round] = []
                self.df = 1

            return parameters_aggregated, metrics_aggregated

        except Exception as e:
            logger.error("aggregate_fit error")
            logger.error("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> list[tuple[ClientProxy, EvaluateIns]]:
        try:
            client_evaluate_list = super().configure_evaluate(server_round, parameters, client_manager)
            logger.info(f"selected clients {self.selected_clients} round {server_round} df {self.df}")
            for i in range(len(client_evaluate_list)):
                client_tuple = client_evaluate_list[i]
                config = client_tuple[1].config
                client_id = client_tuple[0].cid
                lt = 0

                if client_id in self.clients_lt:
                    lt = self.clients_lt[client_id]
                config["nt"] = server_round - lt
                config["lt"] = lt
                # logger.info(f"evaluating client {client_id} round {server_round} lt {lt}")
                client_evaluate_list[i][1].config = config
                client_evaluate_list[i][1].parameters = ndarrays_to_parameters([])
            logger.info(f"model shape: {self.model_shape} path {self.file_path} {len(parameters_to_ndarrays(client_evaluate_list[0][1].parameters))}")
            client_evaluate_list = fedpredict_server(global_model_parameters=parameters_to_ndarrays(parameters),
                                     client_evaluate_list=client_evaluate_list, df=self.df, t=server_round,
                                     T=self.number_of_rounds, compression=self.compression, fl_framework="flwr")
            # for i in range(len(client_evaluate_list)):
            #     client_evaluate_list[i][1].parameters = ndarrays_to_parameters([])
            #     client_evaluate_list[i][1].config = {"t": client_evaluate_list[i][1].config["t"], "nt": client_evaluate_list[i][1].config["nt"], "lt": client_evaluate_list[i][1].config["lt"]}
            # logger.info(f"configure_evaluate: client_evaluate_list {len(client_evaluate_list)} parameters {client_evaluate_list[0][1].parameters}, config {client_evaluate_list[0][1].config.keys()}")
            # exit()
            # for client in r:
            #     logger.info(f"depo type client {type(client)} type 0 {type(client[0])} type 1 {type(client[1])}")
            #     logger.info(f"depo parameters {type(client[1].parameters)} config {client[1].config}")
            # for i in range(len(client_evaluate_list)):
            #     p = ndarrays_to_parameters(r[i]["parameters"])
            #     client_evaluate_list[i][1].parameters = p
            #     client_evaluate_list[i][1].config = r[i]["config"]
                # logger.info(f" 1fp: {type(client_original) == type(client_novo)}")
                # logger.info(f" 2fp: {type(client_original[0]) == type(client_novo[0])}")
                # logger.info(f" 3fp: {type(client_original[1]) == type(client_novo[1])}")
                # logger.info(f" 4fp: {type(parameters_to_ndarrays(client_original[1].parameters)) == type(parameters_to_ndarrays(client_novo[1].parameters))}")
                # logger.info(f" 5fp: {type(client_original[1].config) == type(client_novo[1].config)}")
            return client_evaluate_list
        except Exception as e:
            logger.error("configure_evaluate error")
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

            compression = self.compression
            if len(compression) > 0:
                compression = "_" + compression
            file_path = result_path + "{}{}.csv".format(algo, compression)

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


            logger.info("File path2: " + file_path)
            logger.info(data)

            return file_path, header, data
        except Exception as e:
            logger.error("get_results error")
            logger.error("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))