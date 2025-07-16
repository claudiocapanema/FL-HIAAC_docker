import copy
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
    FitRes,
    EvaluateRes,
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
class MultiFedLR(MultiFedAvg):
    """MultiFedR strategy.

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
            super().__init__(args=args, fraction_fit=fraction_fit, fraction_evaluate=fraction_evaluate,
                             min_fit_clients=min_fit_clients, min_evaluate_clients=min_evaluate_clients,
                             min_available_clients=min_available_clients, evaluate_fn=evaluate_fn,
                             on_fit_config_fn=on_fit_config_fn, on_evaluate_config_fn=on_evaluate_config_fn,
                             accept_failures=accept_failures, initial_parameters=initial_parameters,
                             fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
                             evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn, inplace=inplace)
            self.fairness_weight = 2
            self.clients_loss_ME = {client_id: {me: 10 for me in range(self.ME)} for client_id in range(1, self.total_clients + 1)}
            self.clients_num_examples_ME = {client_id: {me: 1 for me in range(self.ME)} for client_id in
                                    range(1, self.total_clients + 1)}
            self.client_id_real_random = {i: None for i in range(self.total_clients + 1)}
            self.beta1 = 0
            self.beta2 = 0
            self.previous_variance = [0 for me in range(self.ME)]
            self.previous_mean_loss = [0 for me in range(self.ME)]
            self.LRcoeff = [0 for me in range(self.ME)]
            self.mean_loss = [0 for me in range(self.ME)]
            self.variance = [0 for me in range(self.ME)]
            self.initialLR = [0 for me in range(self.ME)]

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
            client_manager.wait_for(self.total_clients, 1000)
            """Configure the next round of training."""
            config = {}
            if self.on_fit_config_fn is not None:
                # Custom fit config function provided
                config = self.on_fit_config_fn(server_round)

            if server_round == 1:
                return super().configure_fit(server_round, parameters, client_manager)

            # Sample clients
            sample_size, min_num_clients = self.num_fit_clients(
                client_manager.num_available()
            )

            n_clients = int(self.total_clients * self.fraction_fit)

            logging.info("""sample clientes {} {} disponiveis {} rodada {} n clients {}""".format(sample_size, min_num_clients, client_manager.num_available(), server_round, n_clients))
            clients = client_manager.sample(
                num_clients=self.total_clients, min_num_clients=self.total_clients
            )
            # clients = [clients[key] for key in list(clients)]
            ids = [i + 1 for i in range(len(clients) + 1)]
            logger.info(f"client ids {ids} n clients {n_clients}")
            ids = np.random.choice(ids, n_clients, replace=False).tolist()

            self.n_trained_clients = len(clients)
            logging.info("""selecionados {} rodada {}""".format(self.n_trained_clients, server_round))
            logger.info(f"mapeamento {self.client_id_real_random}")

            selected_clients_m = [[] for i in range(self.ME)]
            selected_clients_m_ids = [[] for i in range(self.ME)]
            selected_clients_m_ids_random = [[] for i in range(self.ME)]

            me = 0
            for client_id in ids[: self.ME*2]:
                me = int(me % self.ME)
                client = clients[client_id]
                selected_clients_m[me].append(client)
                selected_clients_m_ids[me].append(client_id)
                selected_clients_m_ids_random[me].append(client.cid)
                me += 1

            for client_id in ids[self.ME*2:]:
                client_p = []
                client = clients[client_id]
                for me in range(self.ME):
                    loss = self.clients_loss_ME[client_id][me]
                    num_examples = self.clients_num_examples_ME[client_id][me]
                    client_p.append(loss * num_examples)
                client_p = np.array(client_p)
                client_p = (np.power(client_p, self.fairness_weight - 1)) / np.sum(client_p)
                client_p = client_p / np.sum(client_p)
                print("probal: ", client_p)
                me = np.random.choice([i for i in range(self.ME)], p=client_p)
                selected_clients_m[me].append(client)
                selected_clients_m_ids[me].append(client_id)
                selected_clients_m_ids_random[me].append(client.cid)

            self.selected_clients_m_ids = selected_clients_m_ids
            self.selected_clients_m_ids_random = selected_clients_m_ids_random

            # Return client/config pairs
            logger.info(f"selected_clients_m_ids {selected_clients_m_ids} rodada {server_round}")
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
            count_results_me = [0 for me in range(self.ME)]

            for i in range(len(results)):
                _, result = results[i]
                me = result.metrics["me"]
                client_id = result.metrics["client_id"]
                train_loss = result.metrics["train_loss"]
                num_examples = result.num_examples

                self.clients_loss_ME[client_id][me] = train_loss
                self.clients_num_examples_ME[client_id][me] = num_examples
                logger.info(f"slect ids random {self.selected_clients_m_ids_random[me]} {count_results_me[me]}")
                self.client_id_real_random[client_id] = self.selected_clients_m_ids_random[me][count_results_me[me]]
                count_results_me[me] += 1
                self.selected_clients_m[me].append(client_id)
            logger.info(f"informacoes de momento {self.selected_clients_m} round {server_round}")

            return super().aggregate_fit(server_round, results, failures)
        except Exception as e:
            logger.error("aggregate_fit error")
            logger.error("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))


    def adaptiveLR_Loss_Based(self, server_round, me, loss):

        '''
        Loss-Based Adaptive FedAVG solution
        '''

        # EMA on the mean
        self.mean_loss[me] = self.previous_mean_loss[me] * self.beta1 + (1 - self.beta1) * loss

        # Initialization Bias correction
        self.mean_loss[me] = self.mean_loss[me] / (1 - pow(self.beta1, server_round + 1))

        # EMA on the Variance
        self.variance[me] = self.previous_variance[me] * self.beta2 + (1 - self.beta2) * (
                    loss - self.previous_mean_loss[me]) * (loss - self.previous_mean_loss[me])

        self.previous_mean_loss[me] = copy.deepcopy(self.mean_loss[me])

        temp = copy.deepcopy(self.previous_variance[me])
        self.previous_variance[me] = copy.deepcopy(self.variance[me])
        # Initialization Bias correction
        self.variance[me] = self.variance[me] / (1 - pow(self.beta2, server_round + 1))

        if server_round < 2:
            r = 1
        else:
            r = np.abs(self.variance[me] / (temp / (1 - pow(self.beta2, server_round))))

        self.LRcoeff = self.LRcoeff * self.beta3 + (1 - self.beta3) * r

        coeff = self.LRcoeff / (1 - pow(self.beta3, server_round + 1))

        ### SERVER SCHEDULER Choosing the decay

        # No Decay
        # coeff = min(self.initialLR, self.initialLR*coeff)

        # Decay of 1/t TIME BASED DECAY as in the convergence analysis of FedAVG
        coeff = min(self.initialLR, (self.initialLR * coeff) / (server_round + 1))

        # Decay of 0.99 per round
        # coeff = min(self.initialLR, self.initialLR*coeff*math.pow(0.99, self.current_round))

        for i in self.clients:
            # i.decayLR(coeff)
            i.setLR(coeff)