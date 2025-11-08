import torch
import math

from communication.tuma_bayesian_denoiser import bayesian_channel_multiplicity_estimation_sampbasedapprox_vectorized, estimate_type_samplingbased_logsumexp_vectorized


def compute_T(Z, n, A, B):
    """Computes covariance vector of residual noise (torch, GPU-compatible)."""
    # Z: (n, F)
    c = torch.diagonal(Z.conj().T @ Z).real / n           # (F,)
    taus = c.reshape(B, A).mean(dim=1)                    # (B,)
    taus = taus.repeat_interleave(A)                      # (B*A,)
    return taus

def centralized_decoder(Y, M, U, nAMPIter, B, A, C_blocks, CH_blocks, nP, log_priors, all_Covs, all_Covs_smaller, device, k_true=None, X_true=None, sigma_w=None, plot_perf=False, print_info=False):
    """
    Centralized Decoder with Multisource AMP and Bayesian Estimation.

    Parameters:
    Y : np.ndarray
        Received signal matrix.
    M : int
        Number of codewords.
    U : int
        Number of zones.
    nAMPIter : int
        Number of AMP iterations.
    B : int
        Number of access points (APs).
    A : int
        Number of antennas per AP.
    Cx, Cy : functions
        Encoding (C) and decoding (C^H) functions.
    nP : float
        Normalized power per transmission.
    priors : list[np.ndarray]
        Priors for each zone.
    all_Covs : np.ndarray
        Covariance matrices for sampling-based approximation (specifically used for multiplicity estimation).
    all_Covs_smaller : np.ndarray
        Smaller covariance matrices for sampling-based approximation (specifically used for channel estimation).
    k_true: np.ndarray
        True multiplicity vector.
    X_true: np.ndarray
        True channel matrix.
    sigma_w: float
        True AWGN noise standard deviation.
    plot_perf: bool
        Boolean variable for plotting the channel and multiplicity estimation performance vs AMP iteration.

    Returns:
    est_k : np.ndarray
        Estimated multiplicity vector.
    est_k_per_zone : list[np.ndarray]
        Estimated multiplicity vectors per zones. 
    """
    # Initialization
    n, F = Y.shape
    P = nP/n
    sqrtnP = torch.sqrt(torch.tensor(nP, device=device))
    Z = Y.clone()  # residual signal
    est_X = torch.zeros((U, M, F), dtype=torch.complex64, device=device)
    if plot_perf:
        tv_dists = []
        channel_est_perfs = []
        channel_est_T_perfs = []
        if X_true is not None:
            perf0 = P * sum(torch.norm(X_true[u] - est_X[u], p='fro')**2 for u in range(U))
            channel_est_perfs.append(perf0.item())

    prev_est_X = None

    for t in range(nAMPIter):
        if print_info:
            print(f"\t\tAMP iteration: {t+1}/{nAMPIter}")
                
        # Residual Covariance
        T = compute_T(Z, n, A, B)

        # AMP Updates
        R = torch.zeros((U, M, F), dtype=torch.complex64, device=device)
        Z_batched = Z.unsqueeze(0).expand(U, *Z.shape)
        R = torch.matmul(CH_blocks, Z_batched) + sqrtnP * est_X
        # Q =  torch.zeros((U, F, F), dtype=torch.complex64, device=device)

        for u in range(U):
            # Qu = torch.zeros((F, F), dtype=torch.complex64, device=device)
            # It needs to be updated!
            #    if withOnsager:
            #        est_X[u][m], est_k, Qum = bayesian_channel_multiplicity_estimation_sampbasedapprox_with_onsager(
            #            R_u[m], all_Covs_smaller[u], T, nP, log_priors[u][m], device=device
            #        )
            #        Qu += Qum
            est_X[u] = bayesian_channel_multiplicity_estimation_sampbasedapprox_vectorized(
                R[u], all_Covs_smaller[u], T, nP, 
                log_priors[u][0], # assume same prior for each message (as we have in FL)
                device=device
            )


        Gamma = torch.matmul(C_blocks, est_X).sum(dim=0)
        #Gamma -= (1/n) * torch.matmul(Z,Q)

        Z = Y - math.sqrt(nP) * Gamma

        # Estimate multiplicity/type
        est_k = estimate_type_samplingbased_logsumexp_vectorized(R, T, all_Covs, M, nP, log_priors, device=device)
        #est_k = estimate_type_samplingbased_logsumexp_vectorized_more(R, T, all_Covs, M, nP, log_priors, device=device)

        if plot_perf:
            # TV Distance
            if k_true is not None:
                k_true_t = k_true.to(device) if isinstance(k_true, torch.Tensor) else torch.tensor(k_true, device=device)
                tv_dist = 0.5 * torch.sum(torch.abs(
                    (k_true_t / k_true_t.sum()) - (est_k / est_k.sum())
                )).item()
                tv_dists.append(tv_dist)

            if X_true is not None:
                perf = P * sum(torch.norm(X_true[u] - est_X[u], p='fro')**2 for u in range(U))
                channel_est_perfs.append(perf.item())
            if sigma_w is not None:
                T_true = sigma_w ** 2
                channel_est_T_perfs.append(torch.sum(torch.abs(T - T_true)).item())
        
        # Convergence check
        if t >= 3 and prev_est_X is not None:
            diff = P * sum(torch.norm(prev_est_X[u] - est_X[u], p='fro')**2 for u in range(U))
            if diff < 1e-7:
                break

        prev_est_X = [x.clone() for x in est_X]

    if plot_perf:
        import matplotlib.pyplot as plt
        if sigma_w is not None:
            channel_est_T_perfs.append(
                torch.sum(torch.abs(compute_T(Z, n, A, B) - (sigma_w**2))).item()
            )

        if print_info:
            print("channel_est_perfs =", channel_est_perfs)
            print("channel_est_T_perfs =", channel_est_T_perfs)

        plt.figure()
        plt.semilogy(range(len(channel_est_perfs)), channel_est_perfs, label="Channel Est")
        plt.semilogy(range(len(channel_est_T_perfs)), channel_est_T_perfs, "--", label="Residual Cov Est")
        plt.legend()
        plt.grid(True)
        plt.xlabel("AMP iteration number")
        plt.ylabel("Multisource AMP performance scores")
        plt.show()

        if tv_dists:
            plt.figure()
            plt.plot(range(1, len(tv_dists) + 1), tv_dists)
            plt.ylabel("TV distance")
            plt.xlabel("AMP iteration number")
            plt.grid(True)
            plt.show()


    est_k_per_zone = {}
    for u in range(U):
        est_k_per_zone[u] = est_k.reshape(U, M)[u]

    est_k_total = est_k.reshape(U, M).sum(dim=0)

    return est_k_total, est_k_per_zone
