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
        super().__init__(fraction_fit=fraction_fit, fraction_evaluate=fraction_evaluate, min_fit_clients=min_fit_clients, min_evaluate_clients=min_evaluate_clients, min_available_clients=min_available_clients, evaluate_fn=evaluate_fn, on_fit_config_fn=on_fit_config_fn, on_evaluate_config_fn=on_evaluate_config_fn, accept_failures=accept_failures, initial_parameters=initial_parameters, fit_metrics_aggregation_fn=fit_metrics_aggregation_fn, evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn, inplace=inplace)

        self.tw = []
        for d in self.dataset:
            self.tw.append({"WISDM-W": args.tw, "WISDM-P": args.tw, "ImageNet": args.tw, "CIFAR10": args.tw,
                            "ImageNet_v2": args.tw, "Gowalla": args.tw}[d])
        self.n_classes = [
            {'EMNIST': 47, 'MNIST': 10, 'CIFAR10': 10, 'GTSRB': 43, 'WISDM-W': 12, 'WISDM-P': 12, 'ImageNet': 15,
             "ImageNet_v2": 15, "Gowalla": 7}[dataset] for dataset in
            self.args.dataset]
        self.tw_range = [0.5, 0.1]
        self.models_semi_convergence_flag = [False] * self.ME
        self.models_semi_convergence_count = [0] * self.ME
        self.training_clients_per_model_per_round = {me: [] for me in range(self.ME)}
        self.rounds_since_last_semi_convergence = {me: 0 for me in range(self.ME)}
        self.unique_count_samples = {me: np.array([0 for i in range(self.n_classes[me])]) for me in range(self.ME)}
        self.models_semi_convergence_rounds_n_clients = {m: [] for m in range(self.M)}
        self.accuracy_gain_models = {me: [] for me in range(self.ME)}
        self.stop_cpd = [False for me in range(self.ME)]
        self.re_per_model = int(args.reduction)
        self.fraction_of_classes = np.zeros((self.ME, self.total_clients))
        self.imbalance_level = np.zeros((self.ME, self.total_clients))
        self.lim = []
        self.free_budget_distribution_factor = args.df

    def configure_fit(
        self, server_round: int, parameters: dict, client_manager: ClientManager
    ) -> list[tuple[ClientProxy, FitIns]]:

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
                config = {"t": server_round, "me": me}
                if type(parameters) is dict:
                    fit_ins = FitIns(parameters[me], config)
                else:
                    fit_ins = FitIns(parameters, config)
                clients_m.append((client, fit_ins))
        return clients_m

    def process(self, t):

        """semi-convergence detection"""
        if t == 1:
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
        print("limites: ", self.lim)
        # exit()
        loss_reduction = [0] * self.ME
        for me in range(self.ME):
            if not self.stop_cpd[me]:
                self.rounds_since_last_semi_convergence[me] += 1

                """Stop CPD"""
                print("Modelo m: ", me)
                print("tw: ", self.tw[me], self.results_test_metrics[me]["Loss"])
                losses = self.results_test_metrics[me]["Loss"][-(self.tw[me] + 1):]
                losses = np.array([losses[i] - losses[i + 1] for i in range(len(losses) - 1)])
                if len(losses) > 0:
                    loss_reduction[me] = losses[-1]
                print("Modelo ", me, " losses: ", losses)
                idxs = np.argwhere(losses < 0)
                # lim = [[0.5, 0.25], [0.35, 0.15]]
                upper = self.lim[me][0]
                lower = self.lim[me][1]
                print("Condição 1: ", len(idxs) <= int(self.tw[me] * self.lim[me][0]), "Condição 2: ",
                      len(idxs) >= int(self.tw[me] * lower))
                print(len(idxs), self.tw[me], upper, lower, int(self.tw[me] * upper), int(self.tw[me] * lower))
                if self.rounds_since_last_semi_convergence[me] >= 4:
                    if len(idxs) <= int(self.tw[me] * upper) and len(idxs) >= int(self.tw[me] * lower):
                        self.rounds_since_last_semi_convergence[me] = 0
                        print("a, remaining_clients_per_model, total_clientsb: ",
                              self.training_clients_per_model_per_round[me])
                        self.models_semi_convergence_rounds_n_clients[me].append({'round': t - 2, 'n_training_clients':
                            self.training_clients_per_model_per_round[me][t - 2]})
                        # more clients are trained for the semi converged model
                        print("treinados na rodada passada: ", me, self.training_clients_per_model_per_round[me][t - 2])

                        if flag:
                            self.models_semi_convergence_flag[me] = True
                            self.models_semi_convergence_count[me] += 1
                            flag = False

                    elif len(idxs) > int(self.tw[me] * upper):
                        self.rounds_since_last_semi_convergence[me] += 1
                        self.models_semi_convergence_count[me] -= 1
                        self.models_semi_convergence_count[me] = max(0, self.models_semi_convergence_count[me])