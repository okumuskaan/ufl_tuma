import torch

from communication.tuma_bayesian_denoiser import bayesian_channel_multiplicity_estimation_sampbasedapprox, bayesian_channel_multiplicity_estimation_sampbasedapprox_with_onsager


# =====================================================================
# Access Point (AP) — Local AMP Decoder
# =====================================================================
class AP:
    def __init__(self, id, A, Cx, Cy, U, M, n, P, nAMPiter, Y,
                 all_covs, all_covs_smaller, priors, log_priors,
                 X_true=None, sigma_w=None, withOnsager=False, device=None):
        self.id = id
        self.A = A
        self.Cx = Cx
        self.Cy = Cy
        self.U = U
        self.M = M
        self.n = n
        self.P = P
        self.nP = n * P
        self.nAMPiter = nAMPiter
        self.all_covs = all_covs.to(device)
        self.all_covs_smaller = all_covs_smaller.to(device)
        self.priors = [p.to(device) for p in priors]
        self.log_priors = [lp.to(device) for lp in log_priors]
        self.Y = Y.to(device)
        self.X_true = [x.to(device) for x in X_true] if X_true is not None else None
        self.sigma_w = sigma_w
        self.withOnsager = withOnsager
        self.device = device or Y.device

    def __str__(self):
        return f"AP {self.id+1}"

    # -------------------------------------------------------------
    # AMP Decoder (local)
    # -------------------------------------------------------------
    def AMP_decoder(self):
        print(f"\t\tAP{self.id+1} AMP decoder ...")
        device = self.device

        self.Z = self.Y.clone()
        self.est_X = [torch.zeros((self.M, self.A), dtype=torch.complex64, device=device)
                      for _ in range(self.U)]

        self.channel_est_perfs = []
        self.channel_est_T_perfs = []

        if self.X_true is not None:
            perf0 = sum(self.P * torch.norm(self.X_true[u] - self.est_X[u], p='fro')**2 for u in range(self.U))
            self.channel_est_perfs.append(perf0.item())

        for t in range(self.nAMPiter):
            print(f"\t\t\tAMPiter = {t+1}/{self.nAMPiter}")

            # Compute residual noise covariance
            taus = torch.mean(torch.diagonal(self.Z.conj().T @ self.Z).real / self.n)
            self.T = torch.ones(self.A, device=device) * taus

            self.Gamma = torch.zeros_like(self.Z, dtype=torch.complex64, device=device)
            self.R = []

            for u in range(self.U):
                # Compute effective observation for zone u
                R_u = self.Cy(self.Z, u) + torch.sqrt(torch.tensor(self.nP, device=device)) * self.est_X[u]
                self.R.append(R_u)

                Qu = torch.zeros((self.A, self.A), dtype=torch.complex64, device=device)
                for m in range(self.M):
                    if self.withOnsager:
                        self.est_X[u][m], _, Qum = bayesian_channel_multiplicity_estimation_sampbasedapprox_with_onsager(
                            R_u[m], self.all_covs_smaller[u], self.T, self.nP, self.log_priors[u][m], device=device
                        )
                        Qu += Qum
                    else:
                        self.est_X[u][m] = bayesian_channel_multiplicity_estimation_sampbasedapprox(
                            R_u[m], self.all_covs_smaller[u], self.T, self.nP, self.log_priors[u][m], device=device
                        )

                self.Gamma += self.Cx(self.est_X[u], u)
                if self.withOnsager:
                    self.Gamma -= (1.0 / self.n) * self.Z @ Qu

            self.Z = self.Y - torch.sqrt(torch.tensor(self.nP, device=device)) * self.Gamma

            # Channel estimation performance
            if self.X_true is not None:
                perf = sum(self.P * torch.norm(self.X_true[u] - self.est_X[u], p='fro')**2 for u in range(self.U))
                self.channel_est_perfs.append(perf.item())
            if self.sigma_w is not None:
                self.channel_est_T_perfs.append(torch.sum(torch.abs(self.T - (self.sigma_w**2))).item())

        # Compute local log-likelihoods for later aggregation
        self.log_likelihoods = self.compute_local_likelihood(self.R, self.T, self.all_covs)

        if self.sigma_w is not None:
            taus = torch.mean(torch.diagonal(self.Z.conj().T @ self.Z).real / self.n)
            self.channel_est_T_perfs.append(torch.sum(torch.abs(torch.ones(self.A, device=device)*taus - (self.sigma_w**2))).item())

    # -------------------------------------------------------------
    # Local log-likelihood computations
    # -------------------------------------------------------------
    def compute_loglikelihoods_with_logsumexptrick(self, y, all_Covs, T, nP, Ns):
        logai = -((torch.abs(y)**2) / (T + nP * all_Covs) + torch.log(torch.pi * (T + nP * all_Covs))).sum(dim=-1) - torch.log(torch.tensor(Ns, device=self.device, dtype=torch.float64))
        maxlogai, _ = torch.max(logai, dim=-1, keepdim=True)
        loglikelihoods = (maxlogai[:, 0] + torch.log(torch.sum(torch.exp(logai - maxlogai), dim=-1)))
        return loglikelihoods

    def compute_local_likelihood(self, R, T, all_Covs):
        loglikelihoods = []
        Kmax = all_Covs.shape[1]
        Ns = all_Covs.shape[-2]
        for u in range(self.U):
            loglikelihoods_u = torch.zeros((self.M, Kmax), dtype=torch.float64, device=self.device)
            for mu in range(self.M):
                loglikelihoods_um = self.compute_loglikelihoods_with_logsumexptrick(R[u][mu], all_Covs[u], T, self.nP, Ns)
                loglikelihoods_u[mu] = loglikelihoods_um
            loglikelihoods.append(loglikelihoods_u)
        return loglikelihoods


# =====================================================================
# CPU — Aggregation and Type Estimation
# =====================================================================
class CPU:
    def __init__(self, Y, U, A, B, all_Covs, all_Covs_smaller,
                 priors, log_priors, Cx, Cy, M, n, P, nAMPiter, Xs_true,
                 sigma_w=None, withOnsager=False, device=None):
        self.A = A
        self.B = B
        self.F = A * B
        self.Y = Y.to(device)
        self.U = U
        self.M = M
        self.log_priors = log_priors
        self.device = device

        # Initialize all APs
        self.APs = []
        for b in range(B):
            Y_b = Y[:, b*A:(b+1)*A]
            allCov_b = all_Covs[:, :, :, b*A:(b+1)*A]
            allCovSmall_b = all_Covs_smaller[:, :, :, b*A:(b+1)*A]
            self.APs.append(
                AP(id=b, A=A, Cx=Cx, Cy=Cy, U=U, M=M, n=n, P=P,
                   nAMPiter=nAMPiter, Y=Y_b, all_covs=allCov_b,
                   all_covs_smaller=allCovSmall_b,
                   priors=priors, log_priors=log_priors,
                   X_true=Xs_true[b], sigma_w=sigma_w,
                   withOnsager=withOnsager, device=device)
            )

    def AMP_decoder(self):
        self.est_Xs = []
        self.channel_est_perfs = []
        self.channel_est_T_perfs = []

        for b in range(self.B):
            self.APs[b].AMP_decoder()
            self.est_Xs.append(self.APs[b].est_X)
            self.channel_est_perfs.append(self.APs[b].channel_est_perfs)
            self.channel_est_T_perfs.append(self.APs[b].channel_est_T_perfs)

    def compute_local_loglikelihoods(self):
        self.log_likelihoods_across_users = [ap.log_likelihoods for ap in self.APs]
        self.log_likelihoods_across_users = torch.stack(
            [torch.stack([torch.stack(lu) for lu in ap.log_likelihoods]) for ap in self.APs],
            dim=0
        )

    def aggregate_and_estimate_types(self):
        self.log_posteriors = torch.sum(self.log_likelihoods_across_users, dim=0) + torch.stack(self.log_priors)
        self.est_k = torch.zeros(self.U * self.M, dtype=torch.int64, device=self.device)
        for u in range(self.U):
            for m in range(self.M):
                self.est_k[u * self.M + m] = torch.argmax(self.log_posteriors[u, m])


# =====================================================================
# Distributed Decoder (Top Level)
# =====================================================================
def distributed_decoder(Y, M, U, nAMPIter, B, A, Cx, Cy, nP, priors, log_priors,
                        all_Covs, all_Covs_smaller, device, withOnsager=False,
                        k_true=None, X_true=None, sigma_w=None, plot_perf=False):
    n, _ = Y.shape
    P = nP / n

    # Split X_true per AP
    X_true_across_APs = []
    for b in range(B):
        sub_X = [X_true[u].reshape(-1, B, A)[:, b, :] for u in range(U)]
        X_true_across_APs.append(sub_X)

    cpu = CPU(Y=Y, U=U, A=A, B=B, all_Covs=all_Covs, all_Covs_smaller=all_Covs_smaller,
              priors=priors, log_priors=log_priors, Cx=Cx, Cy=Cy, M=M, n=n, P=P,
              nAMPiter=nAMPIter, Xs_true=X_true_across_APs, sigma_w=sigma_w,
              withOnsager=withOnsager, device=device)

    cpu.AMP_decoder()

    channel_est_perfs = torch.tensor(cpu.channel_est_perfs, device=device).sum(dim=0)
    channel_est_T_perfs = torch.tensor(cpu.channel_est_T_perfs, device=device).sum(dim=0)

    if plot_perf:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.semilogy(channel_est_perfs.cpu(), label="est_ch")
        plt.semilogy(channel_est_T_perfs.cpu(), label="est_T")
        plt.legend()
        plt.show()

    cpu.compute_local_loglikelihoods()
    cpu.aggregate_and_estimate_types()

    est_k_per_zone = {u: cpu.est_k.reshape(-1, M)[u] for u in range(U)}
    return cpu.est_k.reshape(-1, M).sum(dim=0), est_k_per_zone
