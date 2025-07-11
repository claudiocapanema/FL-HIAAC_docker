from flwr.common import (
    EvaluateIns,
    parameters_to_ndarrays
)
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from itertools import islice
from fedpredict import fedpredict_server, fedpredict_layerwise_similarity
from fedpredict.utils.compression_methods.sparsification import sparse_matrix, sparse_bytes

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
            self.compression = args.compression
            self.similarity_list_per_layer = {}
            self.initial_similarity = 0
            self.current_similarity = 0
            self.model_shape_mefl = None
            self.similarity_between_layers_per_round = {}
            self.similarity_between_layers_per_round_and_client = {}
            self.mean_similarity_per_round = {}
            self.df = 0
            self.compressed_size = 0
            self.test_metrics_names += ["df", "Model size (compressed)"]
            self.train_metrics_names += ["df", "Model size (compressed)"]
            self.results_test_metrics = {metric: [] for metric in self.test_metrics_names}
            self.results_test_metrics_w = {metric: [] for metric in self.test_metrics_names}
            self.clients_results_test_metrics = {metric: [] for metric in self.test_metrics_names}
            #self.file_path = None
        except Exception as e:
            logger.error("__init__ error")
            logger.error("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> list[tuple[ClientProxy, FitIns]]:
        try:
            self.previous_global_parameters = parameters_to_ndarrays(parameters)
            return super().configure_fit(server_round, parameters, client_manager)
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

            clients_parameters_update = []
            for i in range(len(results)):
                _, result = results[i]
                client_id = self.selected_clients[i]
                lt = result.metrics["lt"]
                self.clients_lt[client_id] = lt
                client_parameters_update = [current - previous for current, previous in
                 zip(parameters_to_ndarrays(result.parameters), self.previous_global_parameters)]
                logger.info(f"foi {len(client_parameters_update)}")
                clients_parameters_update.append(client_parameters_update)

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
            if server_round == 2:
                flag = True

            if "dls" in self.compression:
                if server_round == 1:
                    self.df = 1
                elif flag and server_round >= 2:
                    # logger.info(f"tipo: {[type(i) for i in parameters_to_ndarrays(parameters_aggregated)]}")
                    # exit()

                    logger.info(f"client update {[len(i) for i in clients_parameters_update]}")
                    global_parameter_update = [current - previous for current, previous in
                                        zip(parameters_to_ndarrays(parameters_aggregated),
                                            self.previous_global_parameters)]
                    self.similarity_between_layers_per_round_and_client[server_round], \
                        self.similarity_between_layers_per_round[server_round], self.mean_similarity_per_round[
                        server_round], self.similarity_list_per_layer, self.df = fedpredict_layerwise_similarity(
                        global_parameter=global_parameter_update, clients_parameters=clients_parameters_update,
                        similarity_per_layer_list=self.similarity_list_per_layer)
                   # self.df = 0
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
                self.df = 0

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
            nts = []
            new_client_evaluate_list = []
            for i in range(len(client_evaluate_list)):
                client_tuple = client_evaluate_list[i]
                config = client_tuple[1].config
                client_id = client_tuple[0].cid
                lt = 0

                if client_id in self.clients_lt:
                    lt = self.clients_lt[client_id]
                nt = server_round - lt
                config["nt"] = nt
                config["lt"] = lt
                nts.append(nt)
                # logger.info(f"evaluating client {client_id} round {server_round} lt {lt}")
                client_evaluate_list[i][1].config = config
                # client_evaluate_list[i][1].parameters = ndarrays_to_parameters([])

                client_dict = {}
                client_dict["client"] = 0
                client_dict["cid"] = client_id
                client_dict["nt"] = nt
                client_dict["lt"] = lt
                client_dict["t"] = server_round
                new_client_evaluate_list.append(
                    (client_tuple[0], EvaluateIns(parameters, client_dict)))


            # logger.info(f"model shape: {self.model_shape} path {self.file_path} {len(parameters_to_ndarrays(client_evaluate_list[0][1].parameters))}")
            logger.info(f"submetidos t: {server_round} T: {self.number_of_rounds} df: {self.df} nts: {nts}")
            client_evaluate_list = fedpredict_server(global_model_parameters=parameters_to_ndarrays(parameters),
                                     client_evaluate_list=new_client_evaluate_list, df=self.df, t=server_round,
                                     T=self.number_of_rounds, compression=self.compression, fl_framework="flwr")
            original_size = sum([j.nbytes for j in parameters_to_ndarrays(parameters)]) * len(client_evaluate_list)
            if self.compression == "sparsification":
                compressed_size = []
                for client in client_evaluate_list:
                    parameters = parameters_to_ndarrays(client[1].parameters)
                    for p in parameters:
                        aux = p[p == 0]
                        # print("quantidade zeros: ", len(aux))
                        sparse = sparse_matrix(p)
                        # print("Tamanho original: ", p.nbytes)
                        b = sparse_bytes(sparse)
                        # print("Apos esparcificacao: ", b)
                        b = min(p.nbytes, b)
                        compressed_size.append(b)
                compressed_size = int(np.mean(compressed_size))
            else:
                compressed_size = int(np.mean([sum([j.nbytes for j in parameters_to_ndarrays(i[1].parameters)]) for i in client_evaluate_list]))
            self.compressed_size = compressed_size
            return client_evaluate_list
        except Exception as e:
            logger.error("configure_evaluate error")
            logger.error("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))

    def add_metrics(self, t, metrics_aggregated):
        try:
            metrics_aggregated["Fraction fit"] = self.fraction_fit
            metrics_aggregated["# training clients"] = self.n_trained_clients
            metrics_aggregated["# available clients"] = self.current_total_clients
            metrics_aggregated["training clients and models"] = self.selected_clients_ids
            metrics_aggregated["Alpha"] = self.alpha
            metrics_aggregated["df"] = self.df
            metrics_aggregated["Model size (compressed)"] = self.compressed_size

            for metric in metrics_aggregated:
                self.results_test_metrics[metric].append(metrics_aggregated[metric])

        except Exception as e:
            logger.error("add_metrics error")
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