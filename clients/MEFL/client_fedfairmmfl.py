import logging

from clients.MEFL.client_multifedavg import ClientMultiFedAvg

logging.basicConfig(level=logging.INFO)  # Configure logging
logger = logging.getLogger(__name__)  # Create logger for the module

class ClientFedFairMMFL(ClientMultiFedAvg):
    def __init__(self, args):
        super(ClientFedFairMMFL, self).__init__(args)