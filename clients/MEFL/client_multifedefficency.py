import logging

import numpy as np
from clients.MEFL.client_multifedavg import ClientMultiFedAvg

logging.basicConfig(level=logging.INFO)  # Configure logging
logger = logging.getLogger(__name__)  # Create logger for the module

class ClientMultiFedEfficiency(ClientMultiFedAvg):
    def __init__(self, args):
        super().__init__(args)
        self.fraction_of_classes = [0 for me in range(self.ME)]
        self.imbalance_level = [0 for me in range(self.ME)]
        self._get_non_iid_degree()


    def _get_non_iid_degree(self):

        for me in range(self.ME):
            train_samples = 0
            y_list = []
            for x, y in self.trainloader[me]:
                train_samples += len(x)
                y_list += list(y)

            train_class_count = {i: 0 for i in range(self.n_classes[me])}
            unique, count = np.unique(y, return_counts=True)
            data_unique_count_dict = dict(zip(unique, count))
            for class_ in data_unique_count_dict:
                train_class_count[class_] = data_unique_count_dict[class_]
            train_class_count = np.array(list(train_class_count.values()))
            threshold = np.sum(train_class_count) / len(train_class_count)
            self.fraction_of_classes[me] = np.count_nonzero(train_class_count) / len(train_class_count)
            self.imbalance_level[me] = len(np.argwhere(train_class_count < threshold)) / len(
                train_class_count)
            logger.info("""fc do cliente {} {} {}""".format(self.client_id, self.fraction_of_classes[me], self.imbalance_level[me]))