import logging
import pickle

import numpy as np
from clients.MEFL.client_multifedavg import ClientMultiFedAvg

logging.basicConfig(level=logging.INFO)  # Configure logging
logger = logging.getLogger(__name__)  # Create logger for the module

class ClientMultiFedEfficiency(ClientMultiFedAvg):
    def __init__(self, args):
        super().__init__(args)
        self.fraction_of_classes = [0 for me in range(self.ME)]
        self.imbalance_level = [0 for me in range(self.ME)]
        self.train_class_count = [0 for me in range(self.ME)]
        self._get_non_iid_degree()

    def evaluate(self, parameters, config):
        """Train the model with data of this client."""

        parameters, dataset_size, tuple_ME = super().evaluate(parameters, config)
        for me in range(self.ME):
            me_str = str(me)
            tuple_me = pickle.loads(tuple_ME[me_str])
            results = tuple_me[2]
            results["fraction_of_classes"] = self.fraction_of_classes
            results["imbalance_level"] = self.imbalance_level
            results["train_class_count"] = self.train_class_count
            results["client_id"] = self.client_id
            tuple_ME[me_str] = pickle.dumps((tuple_me[0], tuple_me[1], results))
        return parameters, dataset_size, tuple_ME


    def _get_non_iid_degree(self):

        for me in range(self.ME):
            train_samples = 0
            y_list = []
            for x, y in self.trainloader[me]:
                train_samples += len(x)
                y_list += list(y)

            self.train_class_count = {i: 0 for i in range(self.n_classes[me])}
            unique, count = np.unique(y, return_counts=True)
            data_unique_count_dict = dict(zip(unique, count))
            for class_ in data_unique_count_dict:
                self.train_class_count[class_] = data_unique_count_dict[class_]
            self.train_class_count = np.array(list(self.train_class_count.values()))
            threshold = np.sum(self.train_class_count) / len(self.train_class_count)
            self.fraction_of_classes[me] = np.count_nonzero(self.train_class_count) / len(self.train_class_count)
            self.imbalance_level[me] = len(np.argwhere(self.train_class_count < threshold)) / len(
                self.train_class_count)
            logger.info("""fc do cliente {} {} {}""".format(self.client_id, self.fraction_of_classes[me], self.imbalance_level[me]))