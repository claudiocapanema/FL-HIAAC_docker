import sys
import copy
import logging
import pickle

import numpy as np
from clients.MEFL.client_multifedavg import ClientMultiFedAvg

logging.basicConfig(level=logging.INFO)  # Configure logging
logger = logging.getLogger(__name__)  # Create logger for the module

class ClientMultiFedLR(ClientMultiFedAvg):
    def __init__(self, args):
        try:
            super().__init__(args)
        except Exception as e:
            logger.error("__init__ error")
            logger.error("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))

    def fit(self, parameters, config):
        """Train the model with data of this client."""
        try:
            # Instruction: get the lr value in the config and update the atribute 'self.optimizer[me]'
            return super().fit(parameters, config)
        except Exception as e:
            logger.error("fit error")
            logger.error("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))