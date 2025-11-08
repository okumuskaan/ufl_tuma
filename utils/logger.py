import pathlib
import numpy as np

class Logger:
    def __init__(self, config_loader):
        self.generate_log_files(config_loader)
        self.test_accs = np.zeros(config_loader.config["federated_learning"]["num_rounds"])
        self.n_selected_clients = np.zeros(config_loader.config["federated_learning"]["num_rounds"])

    def generate_log_files(self, config_loader):
        fl_config = config_loader.get_fl_config()
        select_config = config_loader.get_select_config()
        data_config = config_loader.get_data_config()
        quant_config = config_loader.get_quant_config()
        comm_config = config_loader.get_comm_config()

        if comm_config["perfect_comm"]:
            if quant_config["apply"]:
                folder_name = "./results/perfect_comm/" + select_config["strategy"] + "/"
                fname = 'Q{}_J{}_lr{:.5f}_bs{}_lE{}_a{:.2f}_seed{}_K{}_Ktar{}_q{:.2f}_d{}.csv'.format(
                    quant_config["Q"], quant_config["J"], 
                    fl_config["client_lr"], fl_config["bs"], fl_config["localE"], data_config["alpha"], fl_config["seed"], fl_config["num_clients"], select_config["Ktar"], fl_config["q"], select_config["d"]
                )
            else:
                folder_name = "./results/perfect_comm_no_vq/" + select_config["strategy"] + "/"
                fname = 'lr{:.5f}_bs{}_lE{}_a{:.2f}_seed{}_K{}_Ktar{}_q{:.2f}_d{}.csv'.format(
                    fl_config["client_lr"], fl_config["bs"], fl_config["localE"], data_config["alpha"], fl_config["seed"], fl_config["num_clients"], select_config["Ktar"], fl_config["q"], select_config["d"]
                )
        else:
            folder_name = "./results/comm/" + select_config["strategy"] + "/"
            if comm_config["decoder_type"] == "AMP-DA":
                if comm_config["perfectCSI"]:
                    fname = 'decoder{}_perfectCSI{}_SNRrx{:.1f}_A{}_N{}_nAMPiter{}_NMC{}_Kmax{}_Q{}_J{}_lr{:.5f}_bs{}_lE{}_a{:.2f}_seed{}_K{}_Ktar{}_q{:.2f}_d{}.csv'.format(
                        comm_config["decoder_type"], comm_config["perfectCSI"], 
                        comm_config["SNR_rx_dB"], 
                        comm_config["A"], comm_config["blocklength"], comm_config["nAMPiter"], comm_config["N_MC"], comm_config["Kmax"],
                        quant_config["Q"], quant_config["J"], 
                        fl_config["client_lr"], fl_config["bs"], fl_config["localE"], data_config["alpha"], fl_config["seed"], fl_config["num_clients"], select_config["Ktar"], fl_config["q"], select_config["d"]
                    )
                else:
                    fname = 'decoder{}_perfectCSI{}_phasemaxpidiv{}_SNRrx{:.1f}_A{}_N{}_nAMPiter{}_NMC{}_Kmax{}_Q{}_J{}_lr{:.5f}_bs{}_lE{}_a{:.2f}_seed{}_K{}_Ktar{}_q{:.2f}_d{}.csv'.format(
                        comm_config["decoder_type"], comm_config["perfectCSI"], comm_config["phase_max_pi_div"],
                        comm_config["SNR_rx_dB"], 
                        comm_config["A"], comm_config["blocklength"], comm_config["nAMPiter"], comm_config["N_MC"], comm_config["Kmax"],
                        quant_config["Q"], quant_config["J"], 
                        fl_config["client_lr"], fl_config["bs"], fl_config["localE"], data_config["alpha"], fl_config["seed"], fl_config["num_clients"], select_config["Ktar"], fl_config["q"], select_config["d"]
                    )
            else:
                fname = 'decoder{}_SNRrx{:.1f}_A{}_N{}_nAMPiter{}_NMC{}_Kmax{}_Q{}_J{}_lr{:.5f}_bs{}_lE{}_a{:.2f}_seed{}_K{}_Ktar{}_q{:.2f}_d{}.csv'.format(
                    comm_config["decoder_type"], comm_config["SNR_rx_dB"], 
                    comm_config["A"], comm_config["blocklength"], comm_config["nAMPiter"], comm_config["N_MC"], comm_config["Kmax"],
                    quant_config["Q"], quant_config["J"], 
                    fl_config["client_lr"], fl_config["bs"], fl_config["localE"], data_config["alpha"], fl_config["seed"], fl_config["num_clients"], select_config["Ktar"], fl_config["q"], select_config["d"]
                )
            
        pathlib.Path(folder_name).mkdir(parents=True, exist_ok=True)
        out_fname = folder_name + fname
        with open(out_fname, 'w+') as f:
            print('Round,test_acc,n_selected_clients', file=f)
        self.out_fname = out_fname

    def log(self, rnd, test_acc, n_selected_clients):
        self.test_accs[rnd] = test_acc
        self.n_selected_clients[rnd] = n_selected_clients
        with open(self.out_fname, '+a') as f:
            print('{rnd},{test_acc:.4f},{n_selected_clients:.4f}'.format(
                    rnd=rnd+1, 
                    test_acc=test_acc, 
                    n_selected_clients=n_selected_clients
                ), 
            file=f)
