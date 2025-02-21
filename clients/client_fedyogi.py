import logging

import flwr as fl

from clients.client_fedavg import Client

logging.basicConfig(level=logging.INFO)  # Configure logging
logger = logging.getLogger(__name__)  # Create logger for the module

class ClientFedYogi(Client):
    def __init__(self, args):
        super().__init__(args)