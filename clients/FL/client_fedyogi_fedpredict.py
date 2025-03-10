import logging

from clients.FL.client_fedavg_fedpredict import ClientFedAvgFP

logging.basicConfig(level=logging.INFO)  # Configure logging
logger = logging.getLogger(__name__)  # Create logger for the module

class ClientFedYogiFP(ClientFedAvgFP):
    def __init__(self, args):
        super().__init__(args)