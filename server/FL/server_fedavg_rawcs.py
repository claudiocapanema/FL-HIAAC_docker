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
import pickle
import json

# Initialize Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_energy_by_completion_time(comp, comm, avg_joules):
    try:
        # Understanding Operational 5G: A First Measurement Study on Its Coverage, Performance and Energy Consumption
        # 55% of 4W (avg 5g watts in the energy consumption graphs)
        # 1W = 1J/s
        # from: https://www.stouchlighting.com/blog/electricity-and-energy-terms-in-led-lighting-j-kw-kwh-lm/w
        AVG_5G_JOULES = 2.22
        comm_joules = AVG_5G_JOULES * comm

        comp_joules = avg_joules * comp

        return comp_joules + comm_joules
    except Exception as e:
        logger.error("get_energy_by_completion_time error")
        logger.error("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))

def update_utility(energy, utility):
    try:
        utility = utility - energy
        return utility
    except Exception as e:
        logger.error("update_utility error")
        logger.error("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))
def init_battery_level(max_battery):
    try:
        perc = random.randint(30, 100)
        utility = battery = max_battery * (perc / 100)

        return battery, utility
    except Exception as e:
        logger.error("init_battery_level error")
        logger.error("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))

def idle_power_deduction(battery, elapsed_time):
    try:
        # from: A Novel Non-invasive Method to Measure Power Consumption on Smartphones (39.27 mA = 0.15 W)
        # from: What can Android mobile app developers do about the energy consumption of machine learning (0.1 W)
        IDLE_CONSUMPTION = 0.1
        consumption = IDLE_CONSUMPTION * elapsed_time

        battery -= consumption

        return battery
    except Exception as e:
        logger.error("idle_power_deduction error")
        logger.error("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))

def get_devices_battery_profiles(num_clients):
    try:
        # Common full mAh batteries
        # Values from commom smartphones specifications
        batteries_mah = random.choices([3500, 4000, 4500, 5000], k=num_clients)

        # Converting to joules
        # https://www.axconnectorlubricant.com/rce/battery-electronics-101.html#faq6
        # V = 3.85, commom value from specifications
        batteries_joules = [mAh * 3.85 * 3.6 for mAh in batteries_mah]

        # from: A Novel Non-invasive Method to Measure Power Consumption on Smartphones (39.27 mA = 0.15 W)
        # from: What can Android mobile app developers do about the energy consumption of machine learning (0.1 W)
        IDLE_CONSUMPTION = 0.1

        # Machine Learning at Facebook: Understanding Inference at the Edge
        # Energy Consumption of Batch and Online Data Stream Learning Models for Smartphone-based Human Activity Recognition
        min_battery = 2.0
        max_battery = 5.0
        battery_interval = max_battery - min_battery

        percentuals = [random.random() for _ in range(num_clients)]
        # 1W = 1J/s
        # from: https://www.stouchlighting.com/blog/electricity-and-energy-terms-in-led-lighting-j-kw-kwh-lm/w
        avg_joules = [IDLE_CONSUMPTION + min_battery + percentual * battery_interval for percentual in percentuals]


        return list(zip(batteries_joules, avg_joules))
    except Exception as e:
        logger.error("get_devices_battery_profiles error")
        logger.error("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))

def weighted_loss_avg(results: list[tuple[int, float]]) -> float:
    """Aggregate evaluation results obtained from multiple clients."""
    num_total_evaluation_examples = sum(num_examples for (num_examples, _) in results)
    weighted_losses = [num_examples * loss for num_examples, loss in results]
    return sum(weighted_losses) / num_total_evaluation_examples

# pylint: disable=line-too-long
class FedAvgRAWCS(FedAvg):
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

        self.transmission_threshold = 0.2
        # self.devices_profile = devices_profile
        if self.fraction_fit == 0.3:
            battery = 0.45  # 0.5
            self.link_quality_lower_lim = 0.5  # lq_min
            self.limit_relationship_max_latency = 0  # pt_max
            level = 'low'
        elif self.fraction_fit == 0.5:
            # self.link_quality_lower_lim = 0.3  # lq_min
            # self.limit_relationship_max_latency = 0.3  # pt_max
            # level = 'medium'
            battery = 0.35  # 0.45
            self.link_quality_lower_lim = 0.05  # lq_min
            self.limit_relationship_max_latency = 0.4  # pt_max
            level = 'medium'
        elif self.fraction_fit == 0.7:
            battery = 0.05
            self.link_quality_lower_lim = 0.01  # lq_min
            self.limit_relationship_max_latency = 7  # pt_max
            level = 'high'
            # args.dataset.lower(),
        self.network_profiles = """./clients_selection_configuration_files/rawcs/sim_1_num_clients_{}_num_rounds_100.pkl""".format(
            self.total_clients)
        self.devices_profile = """./clients_selection_configuration_files/rawcs/profiles_sim_cifar10_seed_1_level_{}_alpha_{}_battery_{}.json""".format(
            level, args.alpha[0], battery)
        # self.devices_profile = """./clients_selection_configuration_files/rawcs/profiles_sim_Cifar10_seed_1.json"""
        # self.sim_idx = sim_idx
        # self.input_shape = input_shape
        self.battery_weight = 0.33
        self.cpu_cost_weight = 0.33
        self.link_prob_weight = 0.33
        self.target_accuracy = 1.0
        self.time_percentile = 95
        self.comp_latency_lim = np.inf
        self.clients_info = {}
        self.clients_last_round = []
        self.max_training_latency = 0.0

    def initialize_rawcs(
            self, client_manager: ClientManager
    ) -> Optional[Parameters]:
        try:
            network_profiles = None

            with open(self.network_profiles, 'rb') as file:
                network_profiles = pickle.load(file)
            logger.info(f"Network profiles loaded from {network_profiles}")
            clients_training_time = []

            with open(self.devices_profile, 'r') as file:
                json_dict = json.load(file)
            # logger.info(json_dict)
            # for key in range(self.total_clients):
            #     self.clients_info[key] = {}
            logger.info(f"all clients: {list(client_manager.all().keys())}")
            for i, key in enumerate(list(client_manager.all().keys())):
                self.clients_info[key] = json_dict[str(i)]
                logger.info(f"conf do client {i}: {json_dict[str(i)]}")
                self.clients_info[key]['perc_budget_10'] = False
                self.clients_info[key]['perc_budget_20'] = False
                self.clients_info[key]['perc_budget_30'] = False
                self.clients_info[key]['perc_budget_40'] = False
                self.clients_info[key]['perc_budget_50'] = False
                self.clients_info[key]['perc_budget_60'] = False
                self.clients_info[key]['perc_budget_70'] = False
                self.clients_info[key]['perc_budget_80'] = False
                self.clients_info[key]['perc_budget_90'] = False
                self.clients_info[key]['perc_budget_100'] = False
                self.clients_info[key]['initial_battery'] = self.clients_info[key]['battery']
                if self.clients_info[key]['total_train_latency'] > self.max_training_latency:
                    self.max_training_latency = self.clients_info[key]['total_train_latency']
                clients_training_time.append(self.clients_info[key]['total_train_latency'])
                self.clients_info[key]['network_profile'] = network_profiles[i]
            logger.info(f"teest {clients_training_time} {self.time_percentile}")
            self.comp_latency_lim = np.percentile(clients_training_time, self.time_percentile)

            self.limit_relationship_max_latency = self.comp_latency_lim / self.max_training_latency

        except Exception as e:
            logger.error("initialize_rawcs error")
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
            config["t"] = server_round
            fit_ins = FitIns(parameters, config)

            if server_round == 1:
                self.model_shape = [i.shape for i in flwr.common.parameters_to_ndarrays(parameters)]
                return super().configure_fit(server_round, parameters, client_manager)
            elif server_round == 2:
                self.initialize_rawcs(client_manager)

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

            if server_round == 1:
                available_clients = list(self.clients_info.keys())
            else:
                available_clients = self.sample_fit()

            print("pre selected: ", available_clients)

            selected_cids = self.filter_clients_to_train_by_predicted_behavior(available_clients, server_round)

            # if server_round > 1:
            clients_new = []
            for client in clients:
                if client.cid in selected_cids:
                    clients_new.append(client)
            clients = clients_new
            # clients = np.random.choice(clients, size=min([n_clients, len(clients)]), replace=False)
            # clients = clients[:int(len(clients) * 0.5)]
            # else:
                # clients = np.random.choice(clients, size=min([n_clients, len(clients)]), replace=False)

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

    def sample_fit(self):
        try:
            selected_cids = []
            clients_with_resources = []

            for client in self.clients_info:
                if self.has_client_resources(client):
                    clients_with_resources.append((client, self.clients_info[client]['accuracy']))

                    client_cost = self.get_cost(self.battery_weight, self.cpu_cost_weight, self.link_prob_weight,
                                                self.clients_info[client]['battery'] /
                                                self.clients_info[client]['max_battery'],
                                                self.clients_info[client][
                                                    'total_train_latency'] / self.max_training_latency,
                                                self.clients_info[client]['trans_prob'],
                                                self.target_accuracy,
                                                self.clients_info[client]['accuracy'])

                    client_benefit = self.get_benefit()

                    if random.random() <= (1 - client_cost / client_benefit):
                        selected_cids.append(client)

            if len(selected_cids) == 0 and len(clients_with_resources) != 0:
                clients_with_resources.sort(key=lambda client: client[1])
                selected_cids = [client[0] for client in clients_with_resources[:round(len(clients_with_resources)
                                                                                       * self.fraction_fit)]]
            if len(selected_cids) == 0:
                clients_with_battery = []

                for client in self.clients_info:
                    if self.clients_info[client]['battery'] - \
                            self.clients_info[client]['delta_train_battery'] >= self.clients_info[client]['min_battery']:
                        clients_with_battery.append((client, self.clients_info[client]['accuracy']))

                clients_with_battery.sort(key=lambda client: client[1])

                selected_cids = [client[0] for client in
                                 clients_with_battery[:round(len(clients_with_battery) * self.fraction_fit)]]

            return selected_cids
        except Exception as e:
            logger.error("sample_fit error")
            logger.error("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))

    def sample_eval(self, client_manager: ClientManager, num_clients):
        return client_manager.sample(num_clients)

    def filter_clients_to_train_by_predicted_behavior(self, selected_cids, server_round):
        try:
            # 1 - Atualiza tempo máximo de processamento
            # 2 - Atualiza bateria consumida pelos clientes em treinamento
            # 3 - Atualiza métricas: energia consumida, desperdício de energia, clientes dreanados, latência total
            # 4 - Identificar clientes que falharão a transmissão devido a alguma instabilidade
            # 5 - Atualizar métrica de consumo desperdiçado por falha do cliente
            # 6 - Atualiza lista de clientes que não completaram o treino por falta de bateria ou instabilidade da rede
            total_train_latency_round = 0.0
            total_energy_consumed = 0.0
            total_wasted_energy = 0.0
            round_depleted_battery_by_train = 0
            round_depleted_battery_total = 0
            round_transpassed_min_battery_level = 0
            max_latency = 0.0
            filtered_by_transmisssion = 0
            clients_to_not_train = []

            for cid in selected_cids:
                comp_latency = self.clients_info[cid]['train_latency']
                comm_latency = self.clients_info[cid]['comm_latency']
                avg_joules = self.clients_info[cid]["avg_joules"]

                client_latency = self.clients_info[cid]["total_train_latency"]
                client_consumed_energy = get_energy_by_completion_time(comp_latency, comm_latency, avg_joules)
                new_battery_value = self.clients_info[cid]['battery'] - client_consumed_energy

                if self.clients_info[cid]['battery'] >= self.clients_info[cid]['min_battery'] and new_battery_value < \
                        self.clients_info[cid]['min_battery']:
                    round_transpassed_min_battery_level += 1

                if new_battery_value < 0:
                    total_energy_consumed += self.clients_info[cid]['battery']
                    total_wasted_energy += self.clients_info[cid]['battery']
                    self.clients_info[cid]['battery'] = 0
                    round_depleted_battery_by_train += 1
                    round_depleted_battery_total += 1
                    clients_to_not_train.append(cid)
                else:
                    total_energy_consumed += client_consumed_energy
                    self.clients_info[cid]['battery'] = new_battery_value

                    if self.clients_info[cid]['network_profile'][server_round - 1] < self.transmission_threshold:
                        clients_to_not_train.append(cid)
                        total_wasted_energy += client_consumed_energy
                        filtered_by_transmisssion += 1
                    else:
                        total_train_latency_round += client_latency

                if client_latency > max_latency and cid not in clients_to_not_train:
                    max_latency = client_latency
            # 7 - Remove de clientes selecionados os que foram drenados pelo treinamento
            print("den: ", clients_to_not_train)
            filtered_selected_cids = list(set(selected_cids).difference(clients_to_not_train))
            # 8 - Calcular consumo em estado de espera
            # 9 - Atualizar bateria de cada cliente
            # 10 - Atualizar clientes que foram drenados sem que seja pelo treino
            for cid in self.clients_info:
                old_battery_level = self.clients_info[cid]['battery']

                if old_battery_level > 0:
                    if cid not in filtered_selected_cids:
                        new_battery_level = idle_power_deduction(old_battery_level, max_latency)
                    else:
                        idle_time = max_latency - (self.clients_info[cid]['total_train_latency'])
                        new_battery_level = idle_power_deduction(old_battery_level, idle_time)

                    if self.clients_info[cid]['battery'] >= self.clients_info[cid]['min_battery'] and new_battery_level < \
                            self.clients_info[cid]['min_battery']:
                        round_transpassed_min_battery_level += 1

                    if new_battery_level <= 0:
                        self.clients_info[cid]['battery'] = 0
                        round_depleted_battery_total += 1
                    else:
                        self.clients_info[cid]['battery'] = new_battery_level

            perc_budget_10, perc_budget_100, perc_budget_20, perc_budget_30, perc_budget_40, perc_budget_50, \
                perc_budget_60, perc_budget_70, perc_budget_80, perc_budget_90, avg_battery_perc, batteries_perc = \
                self.transpassed_budget()

            # filename = self.results_dir + self.__repr__() + "_" + self.sim_id + "_" + str(
            #     self.sim_idx) + f"_system_metrics_frac_{self.fraction_fit}_weights_{self.battery_weight}_{self.cpu_cost_weight}_{self.link_prob_weight}.csv"
            #
            # os.makedirs(os.path.dirname(filename), exist_ok=True)
            #
            # with open(filename, 'a') as log:
            #     log.write(f"{server_round},{total_train_latency_round},{total_energy_consumed},{total_wasted_energy},"
            #               f"{len(selected_cids)},{round_depleted_battery_by_train},{round_depleted_battery_total},"
            #               f"{filtered_by_transmisssion},{len(filtered_selected_cids)},"
            #               f"{round_transpassed_min_battery_level},{perc_budget_10},{perc_budget_20},{perc_budget_30},"
            #               f"{perc_budget_40},{perc_budget_50},{perc_budget_60},{perc_budget_70},{perc_budget_80},"
            #               f"{perc_budget_90},{perc_budget_100},{avg_battery_perc}\n"
            #               )
            #
            # filename = self.results_dir + self.__repr__() + "_" + self.sim_id + "_" + str(
            #     self.sim_idx) + f"_batteries_per_client_frac_{self.fraction_fit}_weights_{self.battery_weight}_{self.cpu_cost_weight}_{self.link_prob_weight}.csv"
            #
            # os.makedirs(os.path.dirname(filename), exist_ok=True)
            #
            # with open(filename, 'a') as log:
            #     write = csv.writer(log)
            #     write.writerow(batteries_perc)

            return [str(i) for i in filtered_selected_cids]
        except Exception as e:
            logger.error("filter_clients_to_train_by_predicted_behavior error")
            logger.error("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))

    def update_sample(self, client_manager, selected_cids):
        try:
            selected_clients = []

            for cid in selected_cids:
                selected_clients.append(client_manager.clients[str(cid)])

            return selected_clients
        except Exception as e:
            logger.error("update_sample error")
            logger.error("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))

    def has_client_resources(self, client_id: int):
        try:
            if (self.clients_info[client_id]['battery'] - self.clients_info[client_id]['delta_train_battery']) \
                    >= self.clients_info[client_id]['min_battery'] and self.clients_info[client_id]['trans_prob'] \
                    >= self.link_quality_lower_lim and self.clients_info[client_id][
                'total_train_latency'] / self.max_training_latency <= self.limit_relationship_max_latency:
                return True

            return False
        except Exception as e:
            logger.error("has_client_resources error")
            logger.error("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))

    def get_cost(self, battery_w: float, cpu_w: float, link_w: float, battery: float, cpu_relation: float,
                 link_qlt: float, target_accuracy: float, client_accuracy: float):
        try:
            return (((battery_w) * (1 - battery)) + ((cpu_w) * (cpu_relation)) + ((link_w) * (1 - link_qlt))) ** (
                    target_accuracy - client_accuracy)
        except Exception as e:
            logger.error("has_client_resources error")
            logger.error("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))

    def get_benefit(self):
        return self.target_accuracy

    def transpassed_budget(self):
        try:
            battery_perc = []
            perc_budget_10 = 0
            perc_budget_20 = 0
            perc_budget_30 = 0
            perc_budget_40 = 0
            perc_budget_50 = 0
            perc_budget_60 = 0
            perc_budget_70 = 0
            perc_budget_80 = 0
            perc_budget_90 = 0
            perc_budget_100 = 0
            for cid in self.clients_info:
                depletion = 1 - self.clients_info[cid]['battery'] / self.clients_info[cid]['initial_battery']
                battery_perc.append(self.clients_info[cid]['battery'] / self.clients_info[cid]['initial_battery'])

                if not self.clients_info[cid]['perc_budget_10'] and depletion > 0.1:
                    self.clients_info[cid]['perc_budget_10'] = True
                if not self.clients_info[cid]['perc_budget_20'] and depletion > 0.2:
                    self.clients_info[cid]['perc_budget_20'] = True
                if not self.clients_info[cid]['perc_budget_30'] and depletion > 0.3:
                    self.clients_info[cid]['perc_budget_30'] = True
                if not self.clients_info[cid]['perc_budget_40'] and depletion > 0.4:
                    self.clients_info[cid]['perc_budget_40'] = True
                if not self.clients_info[cid]['perc_budget_50'] and depletion > 0.5:
                    self.clients_info[cid]['perc_budget_50'] = True
                if not self.clients_info[cid]['perc_budget_60'] and depletion > 0.6:
                    self.clients_info[cid]['perc_budget_60'] = True
                if not self.clients_info[cid]['perc_budget_70'] and depletion > 0.7:
                    self.clients_info[cid]['perc_budget_70'] = True
                if not self.clients_info[cid]['perc_budget_80'] and depletion > 0.8:
                    self.clients_info[cid]['perc_budget_80'] = True
                if not self.clients_info[cid]['perc_budget_90'] and depletion > 0.9:
                    self.clients_info[cid]['perc_budget_90'] = True
                if not self.clients_info[cid]['perc_budget_100'] and depletion == 1.0:
                    self.clients_info[cid]['perc_budget_100'] = True

                if self.clients_info[cid]['perc_budget_10']:
                    perc_budget_10 += 1
                if self.clients_info[cid]['perc_budget_20']:
                    perc_budget_20 += 1
                if self.clients_info[cid]['perc_budget_30']:
                    perc_budget_30 += 1
                if self.clients_info[cid]['perc_budget_40']:
                    perc_budget_40 += 1
                if self.clients_info[cid]['perc_budget_50']:
                    perc_budget_50 += 1
                if self.clients_info[cid]['perc_budget_60']:
                    perc_budget_60 += 1
                if self.clients_info[cid]['perc_budget_70']:
                    perc_budget_70 += 1
                if self.clients_info[cid]['perc_budget_80']:
                    perc_budget_80 += 1
                if self.clients_info[cid]['perc_budget_90']:
                    perc_budget_90 += 1
                if self.clients_info[cid]['perc_budget_100']:
                    perc_budget_100 += 1
            return perc_budget_10, perc_budget_100, perc_budget_20, perc_budget_30, perc_budget_40, perc_budget_50, \
                perc_budget_60, perc_budget_70, perc_budget_80, perc_budget_90, sum(battery_perc) / len(battery_perc), \
                battery_perc
        except Exception as e:
            logger.error("transpassed_budget error")
            logger.error("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))