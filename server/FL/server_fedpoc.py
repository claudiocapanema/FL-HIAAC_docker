import torch
import numpy as np
import random
from itertools import islice

import logging

from typing import Callable, Optional

from flwr.common import (
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar
)
from server.FL.server_fedavg import FedAvg
from flwr.common.logger import log
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy

import sys
import flwr

from logging import WARNING
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

# Initialize Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def weighted_loss_avg(results: list[tuple[int, float]]) -> float:
    """Aggregate evaluation results obtained from multiple clients."""
    num_total_evaluation_examples = sum(num_examples for (num_examples, _) in results)
    weighted_losses = [num_examples * loss for num_examples, loss in results]
    return sum(weighted_losses) / num_total_evaluation_examples

# pylint: disable=line-too-long
class FedAvgPOC(FedAvg):
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
        super().__init__(args=args, fraction_fit=fraction_fit, fraction_evaluate=fraction_evaluate, min_fit_clients=min_fit_clients, min_evaluate_clients=min_evaluate_clients, min_available_clients=min_available_clients, evaluate_fn=evaluate_fn, on_fit_config_fn=on_fit_config_fn, on_evaluate_config_fn=on_evaluate_config_fn, accept_failures=accept_failures, initial_parameters=initial_parameters, fit_metrics_aggregation_fn=fit_metrics_aggregation_fn, evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn, inplace=inplace)
        self.clients_train_loss = {}
        self.clients_ids = []

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
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
            if server_round > 1:
                top_k = sorted(self.clients_train_loss.items(), key=lambda x: x[1], reverse=True)[:min([n_clients, len(clients)])]
                top_k = [key for key, value in top_k]
                clients_new = []
                for client in clients:
                    if client.cid in top_k:
                        clients_new.append(client)
                clients = clients_new
            else:
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

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> list[tuple[ClientProxy, EvaluateIns]]:
        try:
            res = super().configure_evaluate(server_round, parameters, client_manager)
            self.clients_ids = []
            for client in res:
                self.clients_ids.append(client[0].cid)
            return res
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
                examples = sum([m["train_samples"] for _, m in eval_metrics])
                train_loss = [(m["train_samples"] * m["train_loss"]) / examples for _, m in eval_metrics]
                for i in range(len(train_loss)):
                    self.clients_train_loss[self.clients_ids[i]] = train_loss[i]
            elif server_round == 1:  # Only log this warning once
                log(WARNING, "No evaluate_metrics_aggregation_fn provided")

            if server_round == 1:
                mode = "w"
            else:
                mode = "w"
            self.add_metrics(server_round, metrics_aggregated)
            self.save_results(mode)


            return loss_aggregated, metrics_aggregated
        except Exception as e:
            logger.error("aggregate_evaluate error")
            logger.error("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))