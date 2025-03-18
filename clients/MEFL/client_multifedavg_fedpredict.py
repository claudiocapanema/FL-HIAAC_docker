import sys
import copy
import json
import pickle
import logging

from clients.MEFL.client_multifedavg import ClientMultiFedAvg
from fedpredict import fedpredict_client_torch

from utils.models_utils import load_model, get_weights, load_data, set_weights, test, train

logging.basicConfig(level=logging.INFO)  # Configure logging
logger = logging.getLogger(__name__)  # Create logger for the module

class ClientMultiFedAvgFedPredict(ClientMultiFedAvg):
    def __init__(self, args):
        super(ClientMultiFedAvgFedPredict, self).__init__(args)
        self.global_model = [None] * self.ME
        for me in range(self.ME):
            # Copy of randomly initialized parameters
            self.global_model = copy.deepcopy(self.model[me])


    def evaluate(self, parameters, config):
        """Evaluate the model on the data this client has."""
        try:
            logger.info("""eval cliente inicio""".format(config))
            t = config["t"]
            parameters = pickle.loads(config["parameters"])
            evaluate_models = json.loads(config["evaluate_models"])
            tuple_me = {}
            logger.info("""modelos para cliente avaliar {} {} {}""".format(evaluate_models, type(parameters), parameters.keys()))
            for me in evaluate_models:
                me = int(me)
                me_str = str(me)
                nt = t - self.lt[me]
                parameters_me = parameters[me_str]
                set_weights(self.model[me], parameters_me)
                combined_model = fedpredict_client_torch(local_model=self.model[me], global_model=self.global_model[me],
                                                         t=t, T=100, nt=nt, device=self.device, fc=1, il=1)
                loss, metrics = test(combined_model, self.valloader[me], self.device, self.client_id, t, self.args.dataset[me], self.n_classes[me])
                metrics["Model size"] = self.models_size[me]
                metrics["Dataset size"] = len(self.valloader[me].dataset)
                metrics["me"] = me
                logger.info("""eval cliente fim {} {}""".format(metrics["me"], metrics))
                tuple_me[me_str] = pickle.dumps((loss, len(self.valloader[me].dataset), metrics))
            return loss, len(self.valloader[me].dataset), tuple_me
        except Exception as e:
            logger.error("evaluate error")
            logger.error("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))

