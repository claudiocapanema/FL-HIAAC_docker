import sys
import logging

from clients.MEFL.client_multifedavg import ClientMultiFedAvg

logging.basicConfig(level=logging.INFO)  # Configure logging
logger = logging.getLogger(__name__)  # Create logger for the module

class ClientMultiFedAvgRR(ClientMultiFedAvg):
    def __init__(self, args):
        try:
            super(ClientMultiFedAvgRR, self).__init__(args)
        except Exception as e:
            logger.error("__init__ error")
            logger.error("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))