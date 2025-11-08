import math
import torch
import torch.multiprocessing as mp
from tqdm import tqdm

from communication.tuma import TUMAEnvironment

import os # <-- added for parallel computation
from communication.tuma import compute_priors, tv_distance # <-- added for parallel computation
from communication.tuma_centralized_decoder import centralized_decoder # <-- added for parallel computation


class CommunicationManager:
    def __init__(
            self,
            selected_clients,
            Ktar,
            topology,
            rnd,
            sub_rnds,
            J,
            SNR_rx_dB,
            decoder_type,
            prior_type,
            N,
            A,
            rho,
            d0, 
            nAMPiter, 
            N_MC, 
            N_MC_smaller, 
            Kmax,
            device,
            perfect_CSI = True, imperfection_model="phase", phase_max_pi_div=6,
            perfect_comm = False,
            plot_type_estimation = False,
            parallel_implementation = False,
            num_workers=16
    ):
        
        M = 2**J
        P = 1 / N

        self.rnd = rnd
        self.sub_rnds = sub_rnds
        self.true_Ka = len(selected_clients)
        self.device = device
        self.perfect_comm = bool(perfect_comm)

        self.plot_type_estimation = plot_type_estimation

        if perfect_comm:
            all_quant_inds = torch.empty((self.true_Ka, self.sub_rnds), dtype=int)
            for idx_sclient, selected_client in enumerate(selected_clients):
                all_quant_inds[idx_sclient] = selected_client.quant_indices
            self.true_quant_indices = []
            self.true_mults = []
            for idx_subrnd in range(self.sub_rnds):
                quant_inds, mults = torch.unique(all_quant_inds[:,idx_subrnd], return_counts=True)
                self.true_quant_indices.append(quant_inds)
                self.true_mults.append(mults)
        else:
            self.parallel_implementation = parallel_implementation
            if parallel_implementation:
                # ⚠️ Switch communication to CPU
                comm_device = torch.device("cpu")
                self.comm_device = comm_device
                self.num_workers = num_workers
                
                self.U = len(topology.zones)
                self.M = M
                self.F = topology.B * A
                self.decoder_type = decoder_type
                self.perfect_CSI = perfect_CSI
                self.imperfection_model = imperfection_model
                self.phase_max = math.pi/phase_max_pi_div
                self.blocklength = N
                self.nP = N * P
                self.nAMPiter = nAMPiter
                self.B = topology.B
                self.A = A

                self.log_priors = load_or_compute_priors(Ktar, prior_type, J, self.U, M, Kmax, comm_device)
                zone_centers = torch.stack([zone.center for zone in topology.zones]).to(comm_device)
                self.all_covs, self.all_covs_smaller = generate_cov_matrices(zone_centers, topology.ap_positions.to(comm_device), topology.side_length, A, self.F, rho, d0, Kmax, N_MC, N_MC_smaller, comm_device)

                self.sigma_w = compute_noise_power(topology.ap_positions.to(comm_device), SNR_rx_dB, P, d0, rho, comm_device)
                self.C_blocks = topology.C_blocks.to(comm_device)
                self.CH_blocks = topology.CH_blocks.to(comm_device)

                self.all_data = torch.vstack([selected_client.quant_indices for selected_client in selected_clients]).to(comm_device)
                self.zone_ids = torch.tensor([selected_client.assigned_zone.id for selected_client in selected_clients], dtype=int, device=comm_device)
                self.LSFCs = torch.vstack([selected_client.LSFCs for selected_client in selected_clients]).to(comm_device)
            else:
                self.TUMA_env = TUMAEnvironment(
                    selected_clients=selected_clients,
                    Ktar            = Ktar,
                    sub_rounds      = sub_rnds,
                    topology        = topology,
                    blocklength     = N,
                    num_antennas    = A,
                    M               = M,
                    path_loss_exp   = rho,
                    ref_distance    = d0,
                    SNR_rx_dB       = SNR_rx_dB,
                    P               = P,
                    prior_type      = prior_type,
                    nAMPiter        = nAMPiter,
                    N_MC            = N_MC,
                    N_MC_smaller    = N_MC_smaller,
                    Kmax            = Kmax,
                    decoder_type    = decoder_type,
                    perfect_CSI     = perfect_CSI,
                    phase_max       = math.pi/phase_max_pi_div,
                    device          = device,
                    perfect_comm    = perfect_comm
                )

    def run_comm(self):
        if self.perfect_comm:
            pass
        else:
            if not self.parallel_implementation:
                for sub_rnd in tqdm(range(self.sub_rnds), desc=f"\tComm. subround"):
                    self.TUMA_env.compute_global_type()
                    self.TUMA_env.generate_X()
                    self.TUMA_env.transmit()
                    self.TUMA_env.decoder()
                    self.TUMA_env.go_next_sub_round()
                    if self.plot_type_estimation:
                        if sub_rnd==0 or sub_rnd==10 or sub_rnd==20:
                            self.TUMA_env.visualize_type_estimation()
            else:
                args = []
                for sub_rnd in range(self.sub_rnds):
                    args.append((
                        self.all_data[:, sub_rnd], self.zone_ids, self.LSFCs, 
                        self.U, self.M, self.B, self.A, self.F, self.nP, 
                        self.C_blocks, self.CH_blocks,
                        self.decoder_type, 
                        self.log_priors, self.all_covs, self.all_covs_smaller,
                        self.blocklength, self.sigma_w, self.nAMPiter, 
                        self.perfect_CSI, self.imperfection_model, self.phase_max,
                        self.comm_device
                    ))

                # Run in parallel
                with mp.get_context("spawn").Pool(processes=self.num_workers, initializer=_init_worker) as pool:
                    results = list(pool.starmap(run_comm_single_round, args))

                # Collect results
                est_quant_indices_list, est_mults_list, est_num_sclients_list = zip(*results)

                # Store results in class attributes
                self.est_quant_indices = list(est_quant_indices_list)
                self.est_mults = list(est_mults_list)
                self.est_num_sclients = list(est_num_sclients_list)

    def return_output(self):
        if self.perfect_comm:
            return self.true_quant_indices, self.true_mults, self.true_Ka
        else:
            if self.parallel_implementation:
                est_quant_indices = [est_quant_ind.to(self.device) for est_quant_ind in self.est_quant_indices]
                est_mults = [est_mult.to(self.device) for est_mult in self.est_mults]
                return est_quant_indices, est_mults, round(sum(self.est_num_sclients)/len(self.est_num_sclients))
            else:
                return self.TUMA_env.est_quant_indices, self.TUMA_env.est_mults, round(sum(self.TUMA_env.est_num_selected_clients)/len(self.TUMA_env.est_num_selected_clients))


# ============================= Functions for parallel computations =============================
def compute_noise_power(ap_positions, SNR_rx_dB, P, ref_distance, path_loss_exp, device):
    """ Computes the noise power based on the received SNR and path loss model. """
    SNR_rx = 10**(SNR_rx_dB / 10)
    min_dist = torch.abs(torch.tensor([(1+1j)*0.0], device=device) - ap_positions,).min()
    SNR_tx = SNR_rx * (1 + (min_dist / ref_distance) ** path_loss_exp)
    return math.sqrt(P / SNR_tx)

def compute_mults_type(data, zone_ids, U, M, device):
    multiplicity_per_zone = torch.zeros(U, M, dtype=int, device=device)
    for u in range(U):
        unique_q_inds, counts = torch.unique(data[zone_ids==u], return_counts=True)
        multiplicity_per_zone[u,unique_q_inds] = counts
    global_k = multiplicity_per_zone.sum(axis=0)
    Ka_true = global_k.sum()
    true_type = global_k/Ka_true
    return multiplicity_per_zone, global_k, Ka_true, true_type

def generate_X(data, multiplicity_per_zone, zone_ids, LSFCs, U, M, F, perfect_CSI, decoder_type, imperfection_model, phase_max, device):
    X = torch.zeros((U, M, F), dtype=torch.complex64, device=device)
    for u in range(U):
        ku = multiplicity_per_zone[u]
        m_s = torch.nonzero(ku)
        for m in m_s:
            ku_m = ku[m]
            xu_m = torch.zeros(F, device=device, dtype=torch.complex64)
            tx_client_inds_um = [
                sclient_ind for sclient_ind, zone_id in enumerate(zone_ids) if zone_id == u and m == data[sclient_ind]
            ]
            h = (torch.randn((ku_m, F), device=device) + 1j * torch.randn((ku_m, F), device=device)) / math.sqrt(2)
            h *= torch.sqrt(LSFCs[tx_client_inds_um])
            if decoder_type == "AMP-DA":
                if perfect_CSI:
                    h_e = h[:,0].reshape(-1,1).clone()
                else:
                    if imperfection_model == "phase":
                        phase = torch.empty((ku_m,1), device=device).uniform_(0, phase_max)
                        h_e = h[:,0].reshape(-1,1).clone() * torch.exp(1j * phase)
                    else:
                        raise ValueError(f"Unknown imperfection_model: {imperfection_model}. Valid option: 'phase'.")
                h /= h_e
                
            X[u,m] = h.sum(axis=0)
    return X

def transmit(X, C_blocks, blocklength, F, sigma_w, nP, device):
    W = (torch.randn(blocklength, F, device=device) + 1j * torch.randn(blocklength, F, device=device)) * math.sqrt(1 / 2) * sigma_w
    Y = math.sqrt(nP) * torch.matmul(C_blocks, X).sum(axis=0) + W
    return Y

def load_or_compute_priors(Ktar, prior_type, J, U, M, Kmax, device):
    """ Loads prior data if available, otherwise computes and saves it. """
    prior_filename = f"./prior_data/priors_Ktar{Ktar}_priortype{prior_type}_J{J}.pt"
    if os.path.exists(prior_filename):
        #print(f"Loading prior data from {prior_filename} ...")
        priors = torch.load(prior_filename).to(device)
        if priors is None:
            raise ValueError(f"{prior_filename} does not contain 'priors' array")
        expected_shape = (U, M, Kmax + 1)
        if priors.shape != expected_shape:
            print(f"Loaded priors shape {priors.shape} != expected {expected_shape}, recomputing.")
            priors = compute_priors(Ktar, M, U, Kmax, prior_type=prior_type, device=device).to(device)
            torch.save(priors, prior_filename)
    else:
        print(f"Computing prior data and saving to {prior_filename} ...")
        priors = compute_priors(Ktar, M, U, Kmax, prior_type=prior_type, device=device).to(device)
        os.makedirs(os.path.dirname(prior_filename), exist_ok=True)
        torch.save(priors, prior_filename)

    # Normalize the truncated priors
    for u in range(U):
        for mu in range(M):
            sum_prob = torch.sum(priors[u, mu, :Kmax+1])
            if sum_prob > 0:
                priors[u, mu, :Kmax+1] /= sum_prob  # Normalize probabilities
            else:
                priors[u, mu, :Kmax+1] = 1.0 / (Kmax + 1)  # Assign uniform distribution if all values are zero
    # Extract priors for each zone and compute log priors
    priors = [priors[u, :, :Kmax+1] for u in range(U)]
    log_priors = [torch.log(priors[u] + 1e-12) for u in range(U)]
    return log_priors

def generate_cov_matrices(zone_centers, ap_positions, side_length, num_antennas, F, path_loss_exp, ref_distance, Kmax, N_MC, N_MC_smaller, device):
    """ Generates covariance matrices used in the decoding process. """
    _, all_covs = generate_all_covs(side_length, ap_positions, zone_centers, num_antennas, F, path_loss_exp, ref_distance, Kmax=Kmax, Ns=N_MC, device=device)
    _, all_covs_smaller = generate_all_covs(side_length, ap_positions, zone_centers, num_antennas, F, path_loss_exp, ref_distance, Kmax=Kmax, Ns=N_MC_smaller, device=device)
    return all_covs, all_covs_smaller

def generate_all_covs(side, nus, zone_centers, A, F, rho, d0, Kmax, Ns, device, num_samples=2000):
    """ Generate covariance matrices for all zones using sampling-based approximation. """
    U = len(zone_centers)
    Qus = generate_uniform_grid(side, num_samples, zone_centers, device)
    positions_for_ks = []
    for k in range(1, Kmax+1):
        pos_k = []
        for u in range(U):
            idx = torch.randint(0, Qus[u].shape[0], (Ns, k), device=device)
            pos_u = Qus[u][idx]
            pos_k.append(pos_u)
        positions_for_ks.append(torch.stack(pos_k, dim=0))  # shape: (U, Ns, k, dim)
    all_Covs_list = []
    for k in range(Kmax):
        covs = generate_cov_matrix(
            positions_for_ks[k], nus, A, device=device, rho=rho, d0=d0
        )
        covs_sum = covs.sum(dim=-2) 
        all_Covs_list.append(covs_sum)
    all_Covs = torch.stack(all_Covs_list, dim=0)  # shape: (Kmax, U, Ns, F)
    zeros_cov = torch.zeros((1, U, Ns, F), device=device)
    all_Covs = torch.cat([zeros_cov, all_Covs], dim=0)  # shape: (Kmax+1, U, Ns, F)
    all_Covs = torch.stack([all_Covs[:, u, :, :] for u in range(U)], dim=0)
    return positions_for_ks, all_Covs

def gamma(q, v=0.0 + 1j * 0.0, rho=3.67, d0=0.01357):
    """ Compute large-scale fading coefficients based on distance. """
    return 1 / (1 + (torch.abs(q - v) / d0) ** rho)

def generate_cov_matrix(pos, nus, A, device, rho=3.67, d0=0.01357):
    """ Generate covariance matrix based on user positions. """
    pos_shape = list(pos.shape)
    return (gamma(pos.unsqueeze(-1).unsqueeze(-2), nus.reshape(-1,1), rho=rho, d0=d0)*torch.ones((1,A),device=device)).reshape(pos_shape + [-1])

def generate_uniform_grid(side, num_points, zone_centers, device, margin=0):
    """ Generate a uniform grid of points for all zones. """
    # Calculate approximate number of points per axis
    num_per_axis = int(math.sqrt(num_points))

    # Generate grid points
    x = torch.linspace(-side/2 + margin, side/2 - margin, num_per_axis, device=device)
    y = torch.linspace(-side/2 + margin, side/2 - margin, num_per_axis, device=device)
    xx, yy = torch.meshgrid(x, y, indexing='xy')
    base_qs = xx.ravel() + 1j * yy.ravel()

    # Generate grid points for each zone
    Qus = torch.zeros([len(zone_centers)] + list(base_qs.shape), device=device)*1j
    for u, zone_center in enumerate(zone_centers):
        Qus[u] = zone_center + base_qs

    return Qus

def decoder(decoder_type, Y, M, U, nAMPiter, B, A, C_blocks, CH_blocks, nP, log_priors, all_covs, all_covs_smaller, global_k, true_type, X, sigma_w, device):
    if decoder_type == "centralized":
        estimated_global_multiplicity, _ = centralized_decoder(
            Y=Y, M=M, U=U, nAMPIter=nAMPiter, B=B, A=A, 
            C_blocks=C_blocks, CH_blocks=CH_blocks, nP=nP, 
            log_priors=log_priors, all_Covs=all_covs, all_Covs_smaller=all_covs_smaller, 
            device=device, 
            k_true=global_k, X_true=X, sigma_w=sigma_w, plot_perf=False)

    est_num_sclients = estimated_global_multiplicity.sum().item()
    estimated_type = estimated_global_multiplicity/(est_num_sclients) if est_num_sclients!=0 else torch.zeros(M, device=device)
    tv_dist = tv_distance(true_type, estimated_type)
    return estimated_global_multiplicity, est_num_sclients, estimated_type, tv_dist

def run_comm_single_round(data, zone_ids, LSFCs, U, M, B, A, F, nP, 
        C_blocks, CH_blocks,
        decoder_type, 
        log_priors, all_covs, all_covs_smaller,
        blocklength, sigma_w, nAMPiter, 
        perfect_CSI, imperfection_model, phase_max,
        device
    ):
    multiplicity_per_zone, global_k, Ka_true, true_type = compute_mults_type(data, zone_ids, U, M, device)
    X = generate_X(data, multiplicity_per_zone, zone_ids, LSFCs, U, M, F, perfect_CSI, decoder_type, imperfection_model, phase_max, device)
    Y = transmit(X, C_blocks, blocklength, F, sigma_w, nP, device)
    estimated_global_multiplicity, est_num_sclients, estimated_type, tv_dist = decoder(decoder_type, Y, M, U, nAMPiter, B, A, C_blocks, CH_blocks, nP, log_priors, all_covs, all_covs_smaller, global_k, true_type, X, sigma_w, device)
    est_quant_inds = torch.nonzero(estimated_global_multiplicity).flatten()
    est_mults = estimated_global_multiplicity[est_quant_inds]
    return est_quant_inds, est_mults, est_num_sclients

def _init_worker():
    import os, torch
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    torch.set_num_threads(1)
