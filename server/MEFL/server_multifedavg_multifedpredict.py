import pickle

from typing import Union

import numpy as np
from flwr.common import (
    EvaluateRes,
)
from flwr.common.logger import log
from flwr.server.strategy.aggregate import aggregate, aggregate_inplace, weighted_loss_avg

from logging import WARNING

import sys
import logging

from typing import Callable, Optional

from flwr.common import (
    FitIns,
    FitRes,
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
class MultiFedAvgMultiFedPredict(MultiFedAvg):
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
        super().__init__(args=args, fraction_fit=fraction_fit, fraction_evaluate=fraction_evaluate, min_fit_clients=min_fit_clients, min_evaluate_clients=min_evaluate_clients, min_available_clients=min_available_clients, evaluate_fn=evaluate_fn, on_fit_config_fn=on_fit_config_fn, on_evaluate_config_fn=on_evaluate_config_fn, accept_failures=accept_failures, initial_parameters=initial_parameters, fit_metrics_aggregation_fn=fit_metrics_aggregation_fn, evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn, inplace=inplace)
        self.need_for_training = [None] * self.ME
        self.min_training_clients_per_model = 3
        self.free_budget = int(self.fraction_fit * self.total_clients) - self.min_training_clients_per_model * self.ME
        self.ME_round_loss = {me: [] for me in range(self.ME)}

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
                """sample clientes {} {} disponiveis {} rodada {} n clients {}""".format(sample_size, min_num_clients,
                                                                                         client_manager.num_available(),
                                                                                         server_round, n_clients))
            clients = client_manager.sample(
                num_clients=n_clients, min_num_clients=n_clients
            )

            n = len(clients) // self.ME
            selected_clients_m = np.array_split(clients, self.ME)
            # train_more_models = []
            # for i, v in enumerate(self.need_for_training):
            #     if v == True:
            #         train_more_models.append(i)
            #
            # if len(train_more_models) > 0:
            #     distributed_budget = self.free_budget // len(train_more_models)
            # else:
            #     distributed_budget = 0
            # training_intensity_me = [self.min_training_clients_per_model] * self.ME
            #
            # for me in train_more_models:
            #     if me in train_more_models:
            #         training_intensity_me[me] += distributed_budget
            #
            # logger.info(f"training intensity me {training_intensity_me} rodada {server_round} free budget {self.free_budget} train more models {train_more_models} need for training {self.need_for_training}")
            # i = 0
            # selected_clients_m = []
            # for me in range(self.ME):
            #     training_intensity = training_intensity_me[me]
            #     j = i + training_intensity
            #     selected_clients_m.append(clients[i:j])
            #     i = j



            self.n_trained_clients = len(clients)
            logging.info("""selecionados {} por modelo {} rodada {}""".format(self.n_trained_clients,
                                                                              [len(i) for i in selected_clients_m],
                                                                              server_round))

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
            fc = {me: [] for me in range(self.ME)}
            il = {me: [] for me in range(self.ME)}
            for i in range(len(results)):
                _, result = results[i]
                me = result.metrics["me"]
                client_id = result.metrics["client_id"]
                fc[me].append(result.metrics["fc"])
                il[me].append(result.metrics["il"])
                self.selected_clients_m[me].append(client_id)
                results_mefl[me].append(results[i])

            for me in range(self.ME):
                fc[me] = np.mean(fc[me])
                il[me] = np.mean(il[me])
                self.need_for_training[me] = False if fc[me] <= 0.94 and il[me] <= 0.4 else True

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

            # Get losses
            for me in range(self.ME):
                self.ME_round_loss[me].append(metrics_aggregated_mefl[me]["Loss"])

            logger.info("""finalizou aggregated fit""")
            logger.info(f"finalizou aggregated fit {server_round}")

            self.calculate_pseudo_t(server_round, self.ME_round_loss[me])

            self.parameters_aggregated_mefl = parameters_aggregated_mefl
            self.metrics_aggregated_mefl = metrics_aggregated_mefl

            return parameters_aggregated_mefl, metrics_aggregated_mefl
        except Exception as e:
            logger.error("aggregate_fit error")
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