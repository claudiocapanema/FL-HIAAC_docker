import logging
import os
import sys

from clients.FL.client_fedavg import Client

from utils.models_utils import get_weights_fedkd, set_weights_fedkd, test_fedkd, train_fedkd, load_model
import torch
from fedpredict.utils.compression_methods.fedkd import fedkd_compression
from fedpredict import layer_compression_range, decompress_global_parameters
from fedpredict.fedpredict_core import layer_compression_range

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

from torch.nn.parameter import Parameter

logging.basicConfig(level=logging.INFO)  # Configure logging
logger = logging.getLogger(__name__)  # Create logger for the module

# Make TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import  sys
import numpy as np
from sklearn.utils.extmath import randomized_svd

# def parameter_svd_write(arrays, n_components_list, svd_type='tsvd'):
#
#     try:
#
#         u = []
#         vt = []
#         sigma_parameters = []
#         arrays_compre = []
#         for i in range(len(arrays)):
#             if type(n_components_list) == list:
#                 n_components = n_components_list[i]
#             else:
#                 n_components = n_components_list
#             # print("Indice da camada: ", i)
#             r = parameter_svd(arrays[i], n_components, svd_type)
#             arrays_compre += r
#
#         return arrays_compre
#
#     except Exception as e:
#         logger.info("paramete_svd")
#         logger.info('Error on line {} {} {}'.format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))
#
# def parameter_svd(layer, n_components, svd_type='tsvd'):
#
#     try:
#         if np.ndim(layer) == 1 or n_components is None:
#             return [layer, np.array([]), np.array([])]
#         elif np.ndim(layer) == 2:
#             r = svd(layer, n_components, svd_type)
#             return r
#         elif np.ndim(layer) >= 3:
#             u = []
#             v = []
#             sig = []
#             for i in range(len(layer)):
#                 r = parameter_svd(layer[i], n_components, svd_type)
#                 u.append(r[0])
#                 v.append(r[1])
#                 sig.append(r[2])
#             return [np.array(u), np.array(v), np.array(sig)]
#
#     except Exception as e:
#         logger.info("parameter_svd")
#         logger.info('Error on line {} {} {}'.format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))
#
#
# def svd(layer, n_components, svd_type='tsvd'):
#
#     try:
#         np.random.seed(0)
#         # print("ola: ", int(len(layer) * n_components), layer.shape, layer)
#         if n_components > 0 and n_components < 1:
#             n_components = int(len(layer) * n_components)
#
#         if svd_type == 'tsvd':
#             U, Sigma, VT = randomized_svd(layer,
#                                           n_components=n_components,
#                                           n_iter=5,
#                                           random_state=0)
#         else:
#             U, Sigma, VT = np.linalg.svd(layer, full_matrices=False)
#             U = U[:, :n_components]
#             Sigma = Sigma[:n_components]
#             VT = VT[:n_components, :]
#
#         # print(U.shape, Sigma.shape, VT.T.shape)
#         return [U, VT, Sigma]
#
#     except Exception as e:
#         logger.info("svd")
#         logger.info('Error on line {} {} {}'.format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))
#
# def if_reduces_size(shape, n_components, dtype=np.float64):
#
#     try:
#         size = np.array([1], dtype=dtype)
#         p = shape[0]
#         q = shape[1]
#         k = n_components
#
#         if p*k + k*k + k*q < p*q:
#             return True
#         else:
#             return False
#
#     except Exception as e:
#         print("svd")
#         logger.info('Error on line {} {} {}'.format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))
#
#
# def inverse_parameter_svd_reading(arrays, model_shape, M=0):
#     try:
#         # print("recebidos aki: ", [i.shape for i in arrays])
#         if M == 0:
#             M = len(model_shape)
#         flag = True
#         for i, j in zip(arrays, model_shape):
#             if i.shape != j:
#                 flag = False
#         if flag:
#             logger.info("nao descomprimiu cliente")
#             return arrays
#         logger.info("descomprimiu cliente")
#         sketched_paramters = []
#         reconstructed_model = []
#         parameter_index = 0
#         sig_ind = 0
#         j = 0
#         for i in range(M):
#             layer_shape = model_shape[i]
#             # print("i32: ", i*3+2)
#             # print("valor i: ", i, i*3, len(model_shape), len(arrays), "valor de M: ", M)
#             u = arrays[i*3]
#             v = arrays[i*3 + 1]
#
#             si = arrays[i*3 + 2]
#             # print("teste", u.shape, v.shape, si.shape, layer_shape)
#             # print("maior: ", i*3 + 2, len(arrays))
#             if len(layer_shape) == 1:
#                 parameter_layer = inverse_parameter_svd(u, v, layer_shape)
#             else:
#                 parameter_layer = inverse_parameter_svd(u, v, layer_shape, si)
#             if parameter_layer is None:
#                 # print("Pos ", i, i*3)
#                 pass
#             reconstructed_model.append(parameter_layer)
#
#         return reconstructed_model
#
#     except Exception as e:
#         logger.info("inverse_paramete_svd")
#         logger.info('Error on line {} {} {}'.format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))
#
#
# def inverse_parameter_svd(u, v, layer_index, sigma=None, sig_ind=None):
#     try:
#         if len(v) == 0:
#             return u
#         if len(layer_index) == 1:
#             # logger.info("u1")
#             return u
#         elif len(layer_index) == 2:
#             # logger.info("u2")
#             return np.matmul(u * sigma, v)
#         elif len(layer_index) == 3:
#             # logger.info("u3")
#             layers_l = []
#             for i in range(len(u)):
#                 layers_l.append(np.matmul(u[i] * sigma[i], v[i]))
#             return np.array(layers_l)
#         elif len(layer_index) == 4:
#             layers_l = []
#             # logger.info("u4")
#             for i in range(len(u)):
#                 layers_j = []
#                 # logger.info("u shape: ", u.shape, " v shape: ", v.shape)
#                 for j in range(len(u[i])):
#                     # logger.info(f"u[i] {u[i]} sigm[i] {sigma[i][j]} v[i] {v[i]}")
#                     layers_j.append(np.matmul(u[i][j] * sigma[i][j], v[i][j]))
#                 layers_l.append(layers_j)
#             return np.array(layers_l)
#
#     except Exception as e:
#         logger.info("inverse_parameter_svd")
#         logger.info('Error on line {} {} {}'.format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))

class ClientFedKD(Client):
    def __init__(self, args):
        try:
            super().__init__(args)
            logger.info("Initializing ClientFedKD")
            self.lr_loss = torch.nn.MSELoss()
            self.round_of_last_fit = 0
            self.rounds_of_fit = 0
            self.accuracy_of_last_round_of_fit = 0
            self.start_server = 0
            # self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9)
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=3e-6)
            feature_dim = 512
            self.W_h = torch.nn.Linear(feature_dim, feature_dim, bias=False)
            self.MSE = torch.nn.MSELoss()
            model_shape = get_weights_fedkd(load_model(args.model[0], args.dataset[0], args.strategy, args.device))
            self.model_shape = [i.shape for i in model_shape]
            self.layers_compression_range = layer_compression_range(self.model_shape)
        except Exception as e:
            logger.info("__init__ error")
            logger.info("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))

    def fit(self, parameters, config):
        """Train the utils with data of this client."""
        try:
            logger.info("""fit cliente fedkd inicio config {} device {}""".format(config, self.device))
            t = config['t']
            self.model.to(self.device)
            if t > 1:
                # logger.info(f"shape original {[i.detach().cpu().numpy().shape for i in self.model.student.parameters()]} \nshape recebido {[i.shape for i in parameters]}")
                logger.info(f"treino cliente {self.client_id} rodada: {t}")
                parameters = [i.detach().cpu().numpy() for i in decompress_global_parameters(parameters, [i.shape for i in get_weights_fedkd(self.model)], self.model.student).to(
                    self.device).parameters()]
                parameters = [Parameter(torch.Tensor(i.tolist())) for i in parameters]
                original_parameters = [i.detach().cpu().numpy() for i in parameters]
            else:
                original_parameters = parameters
            logger.info(f"client {self.client_id} treino rodada {t} parametros shape: {[i.shape for i in parameters]}")
            if len(parameters) > 0:
                set_weights_fedkd(self.model, parameters)
            results = train_fedkd(
                self.model,
                self.trainloader,
                self.valloader,
                self.local_epochs,
                self.lr,
                self.device,
                self.client_id,
                t,
                self.dataset,
                self.n_classes
            )
            self.models_size = self._get_models_size(parameters)
            results["Model size"] = self.models_size
            logger.info("fim fedkd client fit {results}")
            parameters = [trained.detach().cpu().numpy() - original for trained, original in
                                  zip(self.model.student.parameters(), original_parameters)]
            # return get_weights_fedkd(self.model), len(self.trainloader.dataset), results
            # n_components_list = []
            # for i in range(len(parameters)):
            #     compression_range = self.layers_compression_range[i]
            #     if compression_range > 0:
            #         compression_range = self.fedkd_formula(t, self.number_of_rounds, compression_range)
            #     else:
            #         compression_range = None
            #     n_components_list.append(compression_range)

            # parameters_to_send = parameter_svd_write(parameters, n_components_list, 'svd')
            # parameters = parameters_to_send
            return parameters, len(self.trainloader.dataset), results

        except Exception as e:
            logger.info("fit")
            logger.info('Error on line {} {} {}'.format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))


    def evaluate(self, parameters, config):
        """Evaluate the utils on the data this client has."""
        try:
            logger.info("""eval cliente inicio""".format(config))
            t = config["t"]
            nt = t - self.lt
            # logger.info(
            #     f"shape original {[i.detach().cpu().numpy().shape for i in self.model.student.parameters()]} \nshape recebido {[i.shape for i in parameters]}")
            logger.info(f"teste cliente {self.client_id} rodada: {t}")
            # parameters = [i.detach().cpu().numpy() for i in
            #               decompress_global_parameters(parameters, [i.shape for i in get_weights_fedkd(self.model)],
            #                                            self.model.student).to(
            #                   self.device).parameters()]
            # parameters = [Parameter(torch.Tensor(i.tolist())) for i in parameters]
            # original_parameters = [i.detach().cpu().numpy() for i in parameters]
            set_weights_fedkd(self.model, parameters)
            loss, metrics = test_fedkd(self.model, self.valloader, self.device, self.client_id, t, self.dataset, self.n_classes)
            self.models_size = self._get_models_size(parameters)
            metrics["Model size"] = self.models_size
            metrics["Alpha"] = self.alpha
            metrics["nt"] = nt
            logger.info("eval cliente fim")
            return loss, len(self.valloader.dataset), metrics
        except Exception as e:
            logger.info("evaluate")
            logger.info('Error on line {} {} {}'.format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))

    # def fedkd_formula(self, server_round, num_rounds, compression_range):
    #
    #     frac = max(1, abs(1 - server_round)) / num_rounds
    #     compression_range = max(round(frac * compression_range), 1)
    #     logger.info(f"compression range: {compression_range} rounds: {server_round}")
    #     return compression_range
    #
    # def compress(self, server_round, parameters):
    #
    #     try:
    #         layers_compression_range = self.layers_compression_range([i.shape for i in parameters])
    #         n_components_list = []
    #         for i in range(len(parameters)):
    #             compression_range = layers_compression_range[i]
    #             if compression_range > 0:
    #                 frac = 1 - server_round / self.number_of_rounds
    #                 compression_range = max(round(frac * compression_range), 1)
    #             else:
    #                 compression_range = None
    #             n_components_list.append(compression_range)
    #
    #         parameters_to_send = parameter_svd_write(parameters, n_components_list)
    #         return parameters_to_send
    #
    #     except Exception as e:
    #         logger.info("compress")
    #         logger.info('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)