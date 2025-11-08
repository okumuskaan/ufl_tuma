import torch
import math
import os

from communication.tuma_centralized_decoder import centralized_decoder
from communication.tuma_distributed_decoder import distributed_decoder
from communication.amp_da import AMP_DA


def compute_priors(Ka, M, U, Kmax, device, prior_type="binomial", eps=1e-12, sparsity_zeta=None):
    """Build priors with output shape (U, M, Kmax+1)."""

    Kmax = int(Kmax)
    priors = torch.zeros((U, M, Kmax + 1), dtype=torch.float64, device=device)

    # Expected number of active clients per zone
    Ka_zone = float(Ka) / float(U)
    lambda_per_message = Ka_zone / float(M)

    # default sparsity for sparse/zero-inflated variants
    if sparsity_zeta is None and ("_sparse" in prior_type or "zero_infl" in prior_type):
        sparsity_zeta = 0.5
    zeta = 0.0 if sparsity_zeta is None else float(sparsity_zeta)
    zeta = max(0.0, min(0.999, zeta))

    # support values k = [0, 1, ..., Kmax]
    k = torch.arange(Kmax + 1, dtype=torch.float64, device=device)

    prior_type = prior_type.lower()

    # --- Binomial-like priors ---
    if prior_type in ("binomial", "binom", "binomial_sparse", "binom_sparse", "default"):
        n_trials = max(0, int(round(Ka_zone)))
        p = torch.tensor(1.0 / M, dtype=torch.float64, device=device)

        # compute log binomial coefficient safely
        log_coeff = (
            torch.lgamma(torch.tensor(float(n_trials + 1), device=device))
            - (torch.lgamma(k + 1.0) + torch.lgamma(torch.tensor(float(n_trials), device=device) - k + 1.0))
        )

        base_pmf = torch.exp(
            log_coeff
            + k * torch.log(p + eps)
            + (n_trials - k) * torch.log(1 - p + eps)
        )
        base_pmf[k > n_trials] = 0.0

        if prior_type == "default":
            # Custom flat prior except keeping P(0) as in binomial
            p0 = base_pmf[0].clone()
            base_pmf.fill_(0.0)
            base_pmf[0] = p0
            base_pmf[1:] = (1 - p0) / (len(k) - 1)

    # --- Poisson-like priors ---
    elif prior_type in ("poisson", "poiss", "tr_poisson", "tr_poiss",
                        "poisson_sparse", "poiss_sparse", "zero_infl_poisson"):
        lam = torch.tensor(lambda_per_message, dtype=torch.float64, device=device)
        log_pmf = k * torch.log(lam + eps) - torch.lgamma(k + 1.0) - lam
        base_pmf = torch.exp(log_pmf)

    # --- Uniform prior ---
    elif prior_type == "uniform":
        base_pmf = torch.ones(Kmax + 1, dtype=torch.float64, device=device) / (Kmax + 1)

    else:
        raise ValueError(f"Unknown prior_type={prior_type}")

    # --- Apply zero inflation or sparsity ---
    if ("_sparse" in prior_type) or ("zero_infl" in prior_type) or (sparsity_zeta is not None):
        base_pmf = (1 - zeta) * base_pmf
        base_pmf[0] += zeta

    # Normalize
    base_pmf = base_pmf / (base_pmf.sum() + eps)

    # Broadcast to all (U, M)
    priors[:] = base_pmf[None, None, :]

    return priors

def tv_distance(true_type, est_type):
    """Computes the total variation (TV) distance between two type distributions."""
    return torch.abs(true_type - est_type).sum()/2

class TUMAEnvironment:
    """
    TUMA Environment: Handles communication, encoding, and decoding.
    - Manages multiplicity computation.
    - Simulates transmissions.
    - Calls centralized or distributed decoders.
    """
    def __init__(self, 
                 selected_clients, Ktar,
                 sub_rounds,
                 topology, blocklength, num_antennas, M,
                 path_loss_exp, ref_distance, SNR_rx_dB, P, nAMPiter, device,
                 N_MC, N_MC_smaller=1, perfect_comm=False, Kmax=8, 
                 decoder_type="centralized",
                 prior_type="tr_binomial",
                 perfect_CSI=True, imperfection_model="phase", phase_max=math.pi/6):

        # Store initialization parameters
        self.selected_clients = selected_clients
        self.Ktar = Ktar
        self.topology = topology
        self.blocklength = blocklength
        self.num_aps = topology.B 
        self.num_antennas = num_antennas
        self.U = topology.U
        self.ap_positions = topology.ap_positions
        self.M = M
        self.J = int(math.log2(M)) 
        self.F = self.num_aps * num_antennas  
        self.path_loss_exp = path_loss_exp
        self.ref_distance = ref_distance
        self.P = P
        self.nP = self.blocklength * self.P
        self.Y = None
        self.nAMPiter = nAMPiter
        self.N_MC = N_MC
        self.N_MC_smaller = N_MC_smaller
        self.perfect_comm = perfect_comm
        self.Kmax = Kmax
        self.decoder_type = decoder_type
        self.perfect_CSI = perfect_CSI
        self.imperfection_model = imperfection_model
        self.phase_max = phase_max
        self.prior_type = prior_type
        self.device = device

        self.C_blocks = self.topology.C_blocks 
        self.CH_blocks = self.topology.CH_blocks 

        self.sub_rounds = sub_rounds
        self.sub_rnd = 0 # sub_rnd index

        if not self.perfect_comm:
            self.sigma_w = self.compute_noise_power(SNR_rx_dB)

        self.est_quant_indices = []
        self.est_mults = []
        self.est_num_selected_clients = []

    def compute_noise_power(self, SNR_rx_dB):
        """ Computes the noise power based on the received SNR and path loss model. """
        SNR_rx = 10**(SNR_rx_dB / 10)
        nus = torch.tensor([ap.position for ap in self.topology.aps], device=self.device)
        min_dist = torch.abs(torch.tensor([(1+1j)*0.0], device=self.device) - nus,).min()
        SNR_tx = SNR_rx * (1 + (min_dist / self.ref_distance) ** self.path_loss_exp)
        return math.sqrt(self.P / SNR_tx)

    def get_codebook_functions(self):
        """ Returns global encoding (Cx) and decoding (Cy) functions for the decoder. """
        C = torch.cat([zone.C for zone in self.topology.zones], dim=1).to(self.device)
        def Cx(x, u=0):
            """ Encoding function: Projects x using the codebook of zone u. """
            start_idx = u * self.M
            end_idx = (u + 1) * self.M
            return C[:, start_idx:end_idx] @ x

        def Cy(y, u=0):
            """ Decoding function: Applies conjugate transpose projection for zone u. """
            start_idx = u * self.M
            end_idx = (u + 1) * self.M
            return C[:, start_idx:end_idx].conj().T @ y

        return Cx, Cy, C

    def compute_multiplicity_per_zone(self):
        """ Computes the multiplicity vector for each zone. """
        multiplicity_per_zone = {}  
        for zone in self.topology.zones:
            ku = torch.zeros(self.M, dtype=int, device=self.device)  
            for client in self.selected_clients:
                if client.assigned_zone.id == zone.id:
                    ku[client.quant_indices[self.sub_rnd]] += 1 
            multiplicity_per_zone[zone.id] = ku  
        self.multiplicity_per_zone = multiplicity_per_zone 

    def compute_global_multiplicity(self):
        """ Computes the global multiplicity vector by summing local multiplicities. """
        self.compute_multiplicity_per_zone()
        global_k = torch.zeros_like(next(iter(self.multiplicity_per_zone.values())), dtype=int, device=self.device)  
        for zone in self.topology.zones:
            global_k += self.multiplicity_per_zone[zone.id] 
        self.global_multiplicity = global_k
        self.Ka_true = torch.sum(self.global_multiplicity)

    def compute_global_type(self):
        """ Computes the global type by normalizing the global multiplicity vector. """
        self.compute_global_multiplicity()
        total_transmissions = torch.sum(self.global_multiplicity)
        if total_transmissions == 0:
            self.true_type = torch.zeros_like(self.global_multiplicity, dtype=float, device=self.device)
            return torch.zeros_like(self.global_multiplicity, dtype=float, device=self.device)  # Avoid division by zero
        self.true_type = self.global_multiplicity / total_transmissions

    def generate_X(self):
        """ Generates the transmission matrix X based on sensor positions and fading. """
        X = torch.zeros((self.U, self.M, self.F), dtype=torch.complex64, device=self.device)
        for zone in self.topology.zones:
            ku = self.multiplicity_per_zone[zone.id]  
            for m, ku_m in enumerate(ku):
                if ku_m > 0:
                    xu_m = torch.zeros(self.F, device=self.device, dtype=torch.complex64)
                    # Get clients that transmit this message
                    transmitting_clients = [
                        client for client in self.selected_clients if client.assigned_zone.id == zone.id and m == client.quant_indices[self.sub_rnd]
                    ]
                    for client in transmitting_clients:
                        # Generate Rayleigh fading channel
                        h = (torch.randn(self.F, device=self.device) + 1j * torch.randn(self.F, device=self.device)) / math.sqrt(2)
                        # Apply path loss
                        h *= torch.sqrt(client.LSFCs)
                        # Channel inversion if using AMP-DA
                        if self.decoder_type == "AMP-DA":
                            if self.perfect_CSI:
                                h_e = h[0].clone()  # Use exact channel vector
                            else:
                                if self.imperfection_model == "phase":
                                    phase = torch.empty(1, device=self.device).uniform_(0, self.phase_max)
                                    h_e = h[0].clone() * torch.exp(1j * phase)
                                    norm_factor = torch.sqrt(torch.abs(h_e) ** 2)
                                    h_e /= norm_factor
                                else:
                                    raise ValueError(f"Unknown imperfection_model: {self.imperfection_model}. Valid option:, 'phase'.")
                            h /= h_e
                        # Accumulate transmission contributions
                        xu_m += h
                    X[zone.id, m] = xu_m.clone()
        self.X = X

    def transmit(self):
        """ Simulates transmission and computes received signal Y. """
        W = (torch.randn(self.blocklength, self.F, device=self.device) + 1j * torch.randn(self.blocklength, self.F, device=self.device)) * math.sqrt(1 / 2) * self.sigma_w
        self.Y = math.sqrt(self.nP) * torch.matmul(self.C_blocks, self.X).sum(axis=0) + W

    def load_or_compute_priors(self):
        """ Loads prior data if available, otherwise computes and saves it. """
        prior_filename = f"./prior_data/priors_Ktar{self.Ktar}_priortype{self.prior_type}_J{self.J}.pt"
        if os.path.exists(prior_filename):
            #print(f"Loading prior data from {prior_filename} ...")
            priors = torch.load(prior_filename).to(self.device)
            if priors is None:
                raise ValueError(f"{prior_filename} does not contain 'priors' array")
            expected_shape = (self.topology.U, self.M, self.Kmax + 1)
            if priors.shape != expected_shape:
                print(f"Loaded priors shape {priors.shape} != expected {expected_shape}, recomputing.")
                priors = compute_priors(self.Ktar, self.M, self.topology.U, self.Kmax, prior_type=self.prior_type, device=self.device).to(self.device)
                torch.save(priors, prior_filename)
        else:
            print(f"Computing prior data and saving to {prior_filename} ...")
            priors = compute_priors(self.Ktar, self.M, self.topology.U, self.Kmax, prior_type=self.prior_type, device=self.device).to(self.device)
            os.makedirs(os.path.dirname(prior_filename), exist_ok=True)
            torch.save(priors, prior_filename)

        # Normalize the truncated priors
        for u in range(self.topology.U):
            for mu in range(self.M):
                sum_prob = torch.sum(priors[u, mu, :self.Kmax+1])
                if sum_prob > 0:
                    priors[u, mu, :self.Kmax+1] /= sum_prob  # Normalize probabilities
                else:
                    priors[u, mu, :self.Kmax+1] = 1.0 / (self.Kmax + 1)  # Assign uniform distribution if all values are zero
        # Extract priors for each zone and compute log priors
        self.priors = [priors[u, :, :self.Kmax+1] for u in range(self.topology.U)]
        self.log_priors = [torch.log(self.priors[u] + 1e-12) for u in range(self.topology.U)]

    def generate_cov_matrices(self):
        """ Generates covariance matrices used in the decoding process. """
        zone_centers = torch.tensor([zone.center for zone in self.topology.zones], device=self.device)
        _, all_covs = self.generate_all_covs(self.topology.side_length, self.ap_positions, zone_centers, self.num_antennas, self.F, self.path_loss_exp, self.ref_distance, Kmax=self.Kmax, Ns=self.N_MC)
        _, all_covs_smaller = self.generate_all_covs(self.topology.side_length, self.ap_positions, zone_centers, self.num_antennas, self.F, self.path_loss_exp, self.ref_distance, Kmax=self.Kmax, Ns=self.N_MC_smaller)
        self.all_covs = all_covs
        self.all_covs_smaller = all_covs_smaller

    def decoder(self, plot_perf=False):
        """ Decodes the received signal Y using either centralized or distributed decoder. """
        if self.Y is None:
            raise ValueError("There is no transmitted signal!")
        if self.perfect_comm:
            # i) perfect communication:
            estimated_global_multiplicity = self.global_multiplicity.copy()
        else:    
            # ii) noisy communication:
            if self.sub_rnd==0:
                self.Cx, self.Cy, self.C = self.get_codebook_functions()  # Get zone-specific encoding/decoding functions
                self.load_or_compute_priors() # Handle prior computation or loading
                self.generate_cov_matrices() # Generate cov matrices
            if self.decoder_type == "centralized":
                estimated_global_multiplicity, estimated_multiplicity_per_zone = centralized_decoder(Y=self.Y, M=self.M, U=self.topology.U, nAMPIter=self.nAMPiter, B=self.num_aps, A=self.num_antennas, C_blocks=self.C_blocks, CH_blocks=self.CH_blocks, nP=self.nP, log_priors=self.log_priors, all_Covs=self.all_covs, all_Covs_smaller=self.all_covs_smaller, device=self.device,
                                            k_true=torch.cat(list(self.multiplicity_per_zone.values())), X_true=self.X, sigma_w=self.sigma_w, plot_perf=plot_perf)
                self.estimated_multiplicity_per_zone = estimated_multiplicity_per_zone
            elif self.decoder_type == "distributed":
                estimated_global_multiplicity, estimated_multiplicity_per_zone = distributed_decoder(Y=self.Y, M=self.M, U=self.topology.U, nAMPIter=self.nAMPiter, B=self.num_aps, A=self.num_antennas, Cx=self.Cx, Cy=self.Cy, nP=self.nP, priors=self.priors, log_priors=self.log_priors, all_Covs=self.all_covs, all_Covs_smaller=self.all_covs_smaller, device=self.device,  
                                            k_true=torch.cat(list(self.multiplicity_per_zone.values())), X_true=self.X, sigma_w=self.sigma_w, plot_perf=plot_perf)
                self.estimated_multiplicity_per_zone = estimated_multiplicity_per_zone
            elif self.decoder_type == "AMP-DA":
                estimated_global_multiplicity = AMP_DA(self.Y.unsqueeze(1), self.C, self.Ka_true, device=self.device, maxIte=50) 
            else:
                raise ValueError("The decoder type must be either 'centralized', 'distributed' or 'AMP-DA'!")
        self.estimated_global_multiplicity = estimated_global_multiplicity
        self.est_num_sclients = torch.sum(self.estimated_global_multiplicity).item()
        self.estimated_type = estimated_global_multiplicity/(estimated_global_multiplicity.sum()) if estimated_global_multiplicity.sum()!=0 else torch.zeros(self.M, device=self.device)
        self.tv_dist = tv_distance(self.true_type, self.estimated_type)

    def compute_multiplicity_statistics(self):
        """ Computes detailed statistics for each zone and globally. """
        zone_stats = {}
        total_unique_messages = 0
        total_transmissions = 0
        for zone in self.topology.zones:
            ku = self.multiplicity_per_zone[zone.id]  # Multiplicity vector for this zone
            unique_messages = torch.count_nonzero(ku)  # Number of unique messages
            total_sensors = torch.sum(ku)  # Total number of transmissions (active sensors in zone)
            zone_stats[zone.id] = {
                "unique_messages": unique_messages.item(),
                "total_sensors": total_sensors.item()
            }
            total_unique_messages += unique_messages
            total_transmissions += total_sensors
        # Global statistics
        global_stats = {
            "total_unique_messages": total_unique_messages.item(),
            "total_transmissions": total_transmissions.item()
        }
        return zone_stats, global_stats

    def visualize_multiplicity_vector(self):
        """ Visualizes the global multiplicity vector across all messages. """
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 5))
        plt.stem(range(len(self.global_multiplicity)), self.global_multiplicity)
        plt.xlabel("Message Index")
        plt.ylabel("Multiplicity")
        plt.title("Global Multiplicity Vector (Number of Times Each Message is Transmitted)")
        plt.grid(True)
        plt.show()

    def visualize_multiplicity_vector_per_zone(self):
        """ Visualizes the global multiplicity vector across all messages. """
        import matplotlib.pyplot as plt
        for zone_id in range(self.topology.U):
            plt.figure(figsize=(10, 5))
            plt.stem(range(len(self.multiplicity_per_zone[zone_id])), self.multiplicity_per_zone[zone_id])
            plt.xlabel("Message Index")
            plt.ylabel("Multiplicity")
            plt.title(f"Multiplicity Vector for Zone {zone_id+1}")
            plt.grid(True)
            plt.show()

    def visualize_type_estimation(self):
        """ Visualizes the true and estimated type together. """
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 5))
        plt.stem(range(len(self.true_type)), self.true_type, "k", label="true")
        plt.stem(range(len(self.estimated_type)), self.estimated_type, "r--", label="est")
        plt.legend()
        plt.xlabel("Message Index")
        plt.ylabel("True and Estimated Type")
        plt.title(f"Type estimation with tv dist:{self.tv_dist:.4f}")
        plt.grid(True)
        plt.show()

    def gamma(self, q, v=0.0 + 1j * 0.0, rho=3.67, d0=0.01357):
        """ Compute large-scale fading coefficients based on distance. """
        return 1 / (1 + (torch.abs(q - v) / d0) ** rho)

    def generate_cov_matrix(self, pos, nus, A, rho=3.67, d0=0.01357):
        """ Generate covariance matrix based on user positions. """
        pos_shape = list(pos.shape)
        return (self.gamma(pos.unsqueeze(-1).unsqueeze(-2), nus.reshape(-1,1), rho=rho, d0=d0)*torch.ones((1,A),device=self.device)).reshape(pos_shape + [-1])

    def generate_uniform_grid(self, side, num_points, zone_centers, margin=0):
        """ Generate a uniform grid of points for all zones. """
        # Calculate approximate number of points per axis
        num_per_axis = int(math.sqrt(num_points))

        # Generate grid points
        x = torch.linspace(-side/2 + margin, side/2 - margin, num_per_axis, device=self.device)
        y = torch.linspace(-side/2 + margin, side/2 - margin, num_per_axis, device=self.device)
        xx, yy = torch.meshgrid(x, y, indexing='xy')
        base_qs = xx.ravel() + 1j * yy.ravel()

        # Generate grid points for each zone
        Qus = torch.zeros([len(zone_centers)] + list(base_qs.shape), device=self.device)*1j
        for u, zone_center in enumerate(zone_centers):
            Qus[u] = zone_center + base_qs

        return Qus

    def generate_all_covs(self, side, nus, zone_centers, A, F, rho, d0, Kmax, Ns, num_samples=2000):
        """ Generate covariance matrices for all zones using sampling-based approximation. """
        U = len(zone_centers)
        Qus = self.generate_uniform_grid(side, num_samples, zone_centers)
        positions_for_ks = []
        for k in range(1, Kmax+1):
            pos_k = []
            for u in range(U):
                idx = torch.randint(0, Qus[u].shape[0], (Ns, k), device=self.device)
                pos_u = Qus[u][idx]
                pos_k.append(pos_u)
            positions_for_ks.append(torch.stack(pos_k, dim=0))  # shape: (U, Ns, k, dim)
        all_Covs_list = []
        for k in range(Kmax):
            covs = self.generate_cov_matrix(
                positions_for_ks[k], nus, A, rho=rho, d0=d0
            )
            covs_sum = covs.sum(dim=-2) 
            all_Covs_list.append(covs_sum)
        all_Covs = torch.stack(all_Covs_list, dim=0)  # shape: (Kmax, U, Ns, F)
        zeros_cov = torch.zeros((1, U, Ns, F), device=self.device)
        all_Covs = torch.cat([zeros_cov, all_Covs], dim=0)  # shape: (Kmax+1, U, Ns, F)
        all_Covs = torch.stack([all_Covs[:, u, :, :] for u in range(U)], dim=0)
        return positions_for_ks, all_Covs

    def go_next_sub_round(self):
        self.sub_rnd += 1
        quant_inds = torch.nonzero(self.estimated_global_multiplicity).flatten()
        self.est_quant_indices.append(quant_inds)
        self.est_mults.append(self.estimated_global_multiplicity[quant_inds])
        self.est_num_selected_clients.append(self.est_num_sclients)
