import torch 
from copy import deepcopy 
from torch.utils.data import Subset 

from data.data_handler import DataHandler
from model.model_handler import modelHandler
from quant.vq_handler import vqHandler
from core.server import Server
from core.client import Client
from communication.topology import NetworkTopology
from utils.logger import Logger


def initialize_network_and_data(config_loader):

    fl_config = config_loader.get_fl_config()
    data_config = config_loader.get_data_config()
    model_config = config_loader.get_model_config()
    quant_config = config_loader.get_quant_config()
    select_config = config_loader.get_select_config()
    comm_config = config_loader.get_comm_config()
    num_clients = fl_config["num_clients"]
    perfect_comm = bool(comm_config["perfect_comm"])

    seed = fl_config["seed"]
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    device = torch.device(config_loader.config["system"]["device"])
    
    data_handler = DataHandler(dataset=data_config["dataset"], data_dir=data_config["data_dir"], alpha=data_config["alpha"], num_clients=num_clients)
    idx_train, _, server_idx_train, _ = data_handler.distribute_data(num_clients=num_clients)
    server_model = modelHandler(architecture=model_config["architecture"], input_shape=data_handler.input_shape, output_dim=data_handler.output_dim)        
    vq = vqHandler(quant_config["apply"], quant_config["method"], quant_config["J"], quant_config["Q"], device=device)

    area_side = 3*comm_config['zone_side']
    x_coords = (torch.rand(num_clients, device=device) - 0.5) * area_side
    y_coords = (torch.rand(num_clients, device=device) - 0.5) * area_side
    client_positions = x_coords + 1j * y_coords

    server = Server(
        model           = deepcopy(server_model),
        lr              = fl_config["server_lr"],
        test_dataset    = data_handler.test_dataset,
        select_config   = select_config,
        vq              = vq,
        data_subset     = Subset(data_handler.train_dataset, server_idx_train),   
        train_lr        = fl_config["client_lr"],
        localE          = fl_config["localE"],
        bs              = fl_config["bs"], 
        device          = device,
        seed            = seed
    )
    
    clients = [
        Client(
            id              = i,
            pos             = client_positions[i],
            model           = server_model,
            data_subset     = Subset(data_handler.train_dataset, idx_train[i]),
            localE          = fl_config["localE"],
            bs              = fl_config["bs"],
            lr              = fl_config["client_lr"],
            q               = fl_config["q"],
            select_config   = select_config,
            num_clients     = num_clients,
            vq              = vq,
            device          = device,
            seed            = seed,
        )
        for i in range(num_clients)
    ]

    topology = NetworkTopology(side_length=comm_config['zone_side'], device=device)
    if not perfect_comm:
        topology.setup_zone_codebooks(comm_config["blocklength"], 2**quant_config["J"])
        topology.assign_zones_to_clients(clients)
        topology.assign_LSFCs_to_clients(clients, comm_config["d0"], comm_config["rho"], comm_config["A"])

    logger = Logger(config_loader)

    return server, clients, topology, data_handler, logger
