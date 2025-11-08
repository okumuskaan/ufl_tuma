from utils.config_loader import ConfigLoader
from setup.initialize import initialize_network_and_data
from core.federated_learning import simulate_ufl

config_path = "./configs/sample_config.json"
config_loader = ConfigLoader(config_path=config_path)
server, clients, topology, data_handler, logger = initialize_network_and_data(config_loader)
simulate_ufl(
    clients,
    server,
    topology,
    data_handler,
    config_loader,
    logger
)
