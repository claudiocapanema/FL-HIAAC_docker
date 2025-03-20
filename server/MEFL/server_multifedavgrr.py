import sys
import logging

import torch
import numpy as np
from typing import List, Tuple
from flwr.common import Metrics

from typing import Callable, Optional

from flwr.common import (
    FitIns,
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
class MultiFedAvgRR(MultiFedAvg):
    """MultiFedAvgRR strategy.

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
        super().__init__(fraction_fit=fraction_fit, fraction_evaluate=fraction_evaluate, min_fit_clients=min_fit_clients, min_evaluate_clients=min_evaluate_clients, min_available_clients=min_available_clients, evaluate_fn=evaluate_fn, on_fit_config_fn=on_fit_config_fn, on_evaluate_config_fn=on_evaluate_config_fn, accept_failures=accept_failures, initial_parameters=initial_parameters, fit_metrics_aggregation_fn=fit_metrics_aggregation_fn, evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn, inplace=inplace)
        self.selected_clients_m = [None] * self.ME
        self.rr_count = 0

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

            logging.info(
                """sample clientes {} {} disponiveis {} rodada {} n clients {}""".format(sample_size, min_num_clients,
                                                                                         client_manager.num_available(),
                                                                                         server_round, n_clients))

            if self.rr_count == 0:
                clients = client_manager.sample(
                    num_clients=n_clients, min_num_clients=n_clients
                )

                n = len(clients) // self.ME
                self.selected_clients_m = np.array_split(clients, self.ME)

            selected_clients_m_rr = [None] * self.ME
            for i, clients_set in enumerate(self.selected_clients_m):
                new_i = int((i + self.rr_count) % self.ME)
                selected_clients_m_rr[new_i] = clients_set
            self.rr_count += 1
            if self.rr_count == self.ME:
                self.rr_count = 0


            self.n_trained_clients = len(clients)
            logging.info("""selecionados {} por modelo {} rodada {}""".format(self.n_trained_clients,
                                                                              [len(i) for i in selected_clients_m_rr],
                                                                              server_round))

            # Return client/config pairs
            clients_m = []
            for me in range(self.ME):
                sc = selected_clients_m_rr[me]
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