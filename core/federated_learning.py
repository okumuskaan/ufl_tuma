from communication.comm_manager import CommunicationManager


def simulate_ufl(
        clients,
        server, 
        topology, 
        data_handler,
        config_loader,
        logger
):

    fl_config = config_loader.get_fl_config()
    quant_config = config_loader.get_quant_config()
    select_config = config_loader.get_select_config()
    comm_config = config_loader.get_comm_config()

    num_clients = fl_config["num_clients"]
    sel_type = select_config["strategy"]

    if not bool(comm_config["perfect_comm"]):
        assert quant_config["apply"] == 1

    if sel_type=="powd":
        from torch.utils.data import Subset, DataLoader
        import numpy as np
        client_train_losses = []
        def evaluate(num_clients, data_handler, idx_train, server):
            local_losses = []
            for i in range(num_clients):
                dataset = data_handler.train_dataset
                indices = idx_train[i]
                datasubset = Subset(dataset, indices=indices)
                dataloader = DataLoader(datasubset,
                    batch_size=len(datasubset),
                    shuffle=True,
                    pin_memory=True)                
                loss_i, acc_i = server.eval(server.model, dataloader, device=server.device)
                local_losses.append(loss_i)
            return local_losses
        
    for idx_FL_rnd in range(fl_config["num_rounds"]):
        print(f"FL_rnd: {idx_FL_rnd+1}/{fl_config['num_rounds']}")

        # Phase 1: Client activation and receive data from server
        for client in clients: 
            client.choose_self_activity()
            if client.active:
                client.receive(server.global_parameters)
                if quant_config['apply']:
                    client.receive_vq_codebook(server.vq_codebook)
                if sel_type=="self":
                    client.receive_th(server.th)

        # Phase 2: Client selection
        if sel_type=="powd":
            active_client_inds = [client.id for client in clients if client.active]
            if client_train_losses == []:
                idxs_selected = np.random.choice(active_client_inds, size=select_config["Ktar"], replace=False)
            else:
                powd = select_config["powd"]
                rnd_idx = np.random.choice(active_client_inds, size=powd, replace=False)
                repval = list(zip([client_train_losses[i] for i in rnd_idx], rnd_idx))
                repval.sort(key=lambda x: x[0], reverse=True)
                rep = list(zip(*repval))
                idxs_selected = rep[1][:int(select_config["Ktar"])]
        else:
            for client in clients:
                if client.active:
                    client.self_select()
            idxs_selected = [client.id for client in clients if client.selected]
        
        # Phase 3: Selected clients do local training + quantization
        for idx_selected in idxs_selected:
            if sel_type=="powd":
                clients[idx_selected].selected = True
                clients[idx_selected].selecteds.append(1.0)
            clients[idx_selected].local_update()
            if quant_config['apply']:
                clients[idx_selected].quantize()
        selected_clients = [clients[idx] for idx in idxs_selected]

        # Phase 4: Communication of data from clients to server
        if quant_config["apply"]:
            comm_manager = CommunicationManager(
                selected_clients=selected_clients,
                Ktar                    = select_config["Ktar"],
                topology                = topology,
                rnd                     = idx_FL_rnd,
                sub_rnds                = server.D,
                J                       = quant_config["J"],
                SNR_rx_dB               = comm_config["SNR_rx_dB"],
                decoder_type            = comm_config["decoder_type"],
                prior_type              = comm_config["prior_type"],
                N                       = comm_config["blocklength"],
                A                       = comm_config["A"], 
                rho                     = comm_config["rho"],
                d0                      = comm_config["d0"],
                nAMPiter                = comm_config["nAMPiter"], 
                N_MC                    = comm_config["N_MC"], 
                N_MC_smaller            = comm_config["N_MC_smaller"], 
                Kmax                    = comm_config["Kmax"],
                perfect_CSI             = bool(comm_config["perfectCSI"]),
                phase_max_pi_div        = comm_config["phase_max_pi_div"],
                device                  = config_loader.config["system"]["device"],
                perfect_comm            = bool(comm_config["perfect_comm"]),
            )
            comm_manager.run_comm()
            rx_quant_indices, rx_mults, est_n_sel_clients = comm_manager.return_output()
            # Phase 5: Data receive at the server and model aggregation 
            server.receive_tuma_outputs(rx_quant_indices, rx_mults, est_n_sel_clients)
        else:
            model_updates = [selected_client.model_update_state for selected_client in selected_clients]
            # Phase 5: Data receive at the server and model aggregation 
            server.receive_perfect_comm_no_vq(model_updates)

        # Phase 6: Test accuracy and loss computation
        _, test_acc = server.evaluate_approx()
        logger.log(idx_FL_rnd, test_acc, len(selected_clients))

        # Phase 7: VQ codebook learning
        if bool(quant_config['apply']):
            server.learn_VQ_codebook()    

        # Phase 8: For self-selection, server updates the threshold. For powd, compute local train losses. 
        if sel_type=="self":
            server.update_th()
        elif sel_type=="powd":
            client_train_losses = evaluate(num_clients, data_handler, data_handler.client_indices_train, server)
