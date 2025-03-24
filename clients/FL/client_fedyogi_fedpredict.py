import sys
import logging

from clients.FL.client_fedavg_fedpredict import ClientFedAvgFP

logging.basicConfig(level=logging.INFO)  # Configure logging
logger = logging.getLogger(__name__)  # Create logger for the module

class ClientFedYogiFP(ClientFedAvgFP):
    def __init__(self, args):
        try:
            super().__init__(args)
        except Exception as e:
            logger.error("__init__ error")
            logger.error("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))