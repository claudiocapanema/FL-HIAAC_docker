import sys
import logging
import pickle

import numpy as np
from clients.MEFL.client_multifedavg import ClientMultiFedAvg

logging.basicConfig(level=logging.INFO)  # Configure logging
logger = logging.getLogger(__name__)  # Create logger for the module

class ClientMultiFedEfficiency(ClientMultiFedAvg):
    def __init__(self, args):
        try:
            super().__init__(args)
            self.fraction_of_classes = [0 for me in range(self.ME)]
            self.imbalance_level = [0 for me in range(self.ME)]
            self.train_class_count = [{i: 0 for i in range(self.n_classes[me])} for me in range(self.ME)]
            self._get_non_iid_degree()
        except Exception as e:
            logger.error("__init__ error")
            logger.error("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))

    def evaluate(self, parameters, config):
        """Train the model with data of this client."""
        try:
            parameters, dataset_size, tuple_ME = super().evaluate(parameters, config)
            for me in range(self.ME):
                me_str = str(me)
                tuple_me = pickle.loads(tuple_ME[me_str])
                t = config["t"]
                results = tuple_me[2]
                results["fraction_of_classes"] = self.fraction_of_classes[me]
                results["imbalance_level"] = self.imbalance_level[me]
                results["train_class_count"] = self.train_class_count[me]
                results["client_id"] = self.client_id
                logger.info("""testou cliente {} rodada {} modelo {}""".format(self.client_id, t, me))
                tuple_ME[me_str] = pickle.dumps((tuple_me[0], tuple_me[1], results))
            return parameters, dataset_size, tuple_ME
        except Exception as e:
            logger.error("evaluate error")
            logger.error("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))

    def _get_non_iid_degree(self):
        try:
            for me in range(self.ME):
                train_samples = 0
                y_list = []
                for batch in self.trainloader[me]:
                    train_samples += len(batch['label'])
                    y_list += batch['label'].detach().cpu().numpy().tolist()

                self.train_class_count[me] = {i: 0 for i in range(self.n_classes[me])}
                unique, count = np.unique(y_list, return_counts=True)
                data_unique_count_dict = dict(zip(np.array(unique).tolist(), np.array(count).tolist()))
                logger.info("""y: {}""".format(y_list[:10]))
                logger.info("""data unique {}""".format(data_unique_count_dict))
                for class_ in data_unique_count_dict:
                    logger.info("""class local {}""".format(class_))
                    self.train_class_count[me][class_] = data_unique_count_dict[class_]
                self.train_class_count[me] = np.array(list(self.train_class_count[me].values()))
                threshold = np.sum(self.train_class_count[me]) / len(self.train_class_count[me])
                self.fraction_of_classes[me] = np.count_nonzero(self.train_class_count[me]) / len(self.train_class_count[me])
                self.imbalance_level[me] = len(np.argwhere(self.train_class_count[me] < threshold)) / len(
                    self.train_class_count[me])
                self.train_class_count[me] = np.array(self.train_class_count[me]).tolist()
                logger.info("""fc do cliente {} {} {} {}""".format(self.client_id, self.fraction_of_classes[me], self.imbalance_level[me], self.train_class_count[me]))
        except Exception as e:
            logger.error("_get_non_iid_degree error")
            logger.error("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))