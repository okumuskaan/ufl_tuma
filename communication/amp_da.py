"""
AMP-DA Implementation

This file contains the AMP-DA algorithm.

References:
- L. Qiao, J. Zhang, and K. B. Letaief, "Massive Digital Over-the-Air Computation for
Communication-Efficient Federated Edge Learning," 
  IEEE Transactions on Wireless Communications, 2024. https://github.com/liqiao19/MD-AirComp
"""

import torch
import scipy.stats as st


def AMP_DA(y, C, Ka, maxIte, device=None, exx=1e-10, damp=0.3):
    """
    Torch-based Approximate Message Passing - Digital Aggregation (AMP-DA).

    Parameters:
    ----------
    y : torch.Tensor (N_RAs x N_dim)
        Received signal matrix.
    C : torch.Tensor (N_RAs x N_UEs)
        Codebook matrix.
    Ka : int
        Number of active users.
    maxIte : int
        Maximum number of AMP iterations.
    device : torch.device
        Device (CPU or GPU).
    exx : float
        Small constant to prevent numerical issues.
    damp : float
        Damping factor.

    Returns:
    --------
    est_k : torch.Tensor
        Estimated multiplicity vector.
    """

    device = device or y.device
    y = y.to(device)
    C = C.to(device)

    N_RAs = C.shape[0]
    N_UEs = C.shape[1]
    N_dim = y.shape[1]
    N_M = y.shape[1]

    # ----- Parameters and initialization -----
    alphabet = torch.arange(0.0, Ka + 1, device=device)
    M = len(alphabet) - 1

    lam = N_RAs / N_UEs
    c = torch.linspace(0.01, 10, 1024)
    # Use scipy for statistical constants (only CPU, scalar ops)
    c_np = c.cpu().numpy()
    rho_np = (1 - 2 * N_UEs * ((1 + c_np ** 2) * st.norm.cdf(-c_np) - c_np * st.norm.pdf(c_np)) / N_RAs) / (
        1 + c_np ** 2 - 2 * ((1 + c_np ** 2) * st.norm.cdf(-c_np) - c_np * st.norm.pdf(c_np))
    )
    rho = torch.tensor(rho_np, device=device)
    alpha = lam * torch.max(rho) * torch.ones((N_UEs, N_dim), device=device)

    # Initialization
    x_hat = torch.ones((N_UEs, N_dim, N_M), device=device, dtype=torch.complex64)
    var_hat = torch.ones_like(x_hat.real)
    V = torch.ones((N_RAs, N_dim, N_M), device=device)
    Z = y.clone()
    V_new = V.clone()
    Z_new = y.clone()

    sigma2 = torch.tensor(100.0, device=device)
    MSE = torch.zeros(maxIte, device=device)
    MSE[0] = 100.0

    hvar = (torch.norm(y)**2 - N_RAs * sigma2) / (N_dim * lam * torch.max(rho) * torch.norm(C)**2)
    hmean = torch.tensor(0.0, device=device, dtype=torch.complex64)

    alpha_new = torch.ones_like(x_hat.real)
    x_hat_new = torch.ones_like(x_hat)
    var_hat_new = torch.ones_like(var_hat)

    hvarnew = torch.zeros(N_M, device=device)
    hmeannew = torch.zeros(N_M, dtype=torch.complex64, device=device)
    sigma2new = torch.zeros(N_M, device=device)

    # ----- AMP iterations -----
    t = 1
    while t < maxIte:
        x_hat_pre = x_hat.clone()

        for i in range(N_M):
            # Compute residuals
            V_new[:, :, i] = torch.abs(C)**2 @ var_hat[:, :, i]
            Z_new[:, :, i] = C @ x_hat[:, :, i] - ((y[:, :, i] - Z[:, :, i]) / (sigma2 + V[:, :, i])) * V_new[:, :, i]

            # Damping
            Z_new[:, :, i] = damp * Z[:, :, i] + (1 - damp) * Z_new[:, :, i]
            V_new[:, :, i] = damp * V[:, :, i] + (1 - damp) * V_new[:, :, i]

            var1 = (torch.abs(C)**2).T @ (1 / (sigma2 + V_new[:, :, i]))
            var2 = C.conj().T @ ((y[:, :, i] - Z_new[:, :, i]) / (sigma2 + V_new[:, :, i]))

            Ri = var2 / var1 + x_hat[:, :, i]
            Vi = 1 / var1

            sigma2new[i] = (
                ((torch.abs(y[:, :, i] - Z_new[:, :, i]) ** 2)
                 / (torch.abs(1 + V_new[:, :, i] / sigma2) ** 2)
                 + sigma2 * V_new[:, :, i] / (V_new[:, :, i] + sigma2)).mean()
            )

            # Case i == 0 → activity estimation
            if i == 0:
                r_s = Ri.unsqueeze(0) - alphabet[:, None, None].to(device)
                exp_input = -(torch.abs(r_s)**2 / Vi)
                exp_input = torch.clamp(exp_input, min=-700, max=700)
                pf8 = torch.exp(exp_input) / Vi / torch.pi
                pf7 = torch.zeros((M + 1, N_UEs, N_dim), device=device)
                pf7[0] = pf8[0] * (1 - alpha)
                pf7[1:] = pf8[1:] * (alpha / M)
                PF7 = pf7.sum(dim=0)
                pf6 = pf7 / PF7

                AAA = alphabet.view(1, -1, 1)
                BBB = pf6.permute(2, 1, 0)
                x_hat_new[:, :, i] = torch.einsum("ijk,ikn->ijn", BBB, AAA).squeeze(-1).T
                alphabet2 = alphabet**2
                AAA2 = alphabet2.view(1, -1, 1)
                var_hat_new[:, :, i] = torch.einsum("ijk,ikn->ijn", BBB, AAA2).squeeze(-1).T - torch.abs(x_hat_new[:, :, i])**2
                alpha_new[:, :, i] = torch.clamp(pf6[1:].sum(dim=0), exx, 1 - exx)

            # Case i > 0 → channel estimation
            else:
                A = (hvar * Vi) / (Vi + hvar)
                B = (hvar * Ri + Vi * hmean) / (Vi + hvar)

                lll = (
                    torch.log(Vi / (Vi + hvar)) / 2
                    + torch.abs(Ri)**2 / (2 * Vi)
                    - torch.abs(Ri - hmean)**2 / (2 * (Vi + hvar))
                )
                pai = torch.clamp(alpha / (alpha + (1 - alpha) * torch.exp(-lll)), exx, 1 - exx)

                x_hat_new[:, :, i] = pai * B
                var_hat_new[:, :, i] = pai * (torch.abs(B)**2 + A) - torch.abs(x_hat_new[:, :, i])**2

                hmeannew[i] = (torch.sum(pai * B, dim=0) / torch.sum(pai, dim=0)).mean()
                hvarnew[i] = (torch.sum(pai * (torch.abs(hmean - B)**2 + Vi), dim=0) / torch.sum(pai, dim=0)).mean()
                alpha_new[:, :, i] = torch.clamp(pai, exx, 1 - exx)

        # Aggregate over antennas
        if N_M > 1:
            hvar = hvarnew[1:].mean()
            hmean = hmeannew[1:].mean()
        sigma2 = sigma2new.mean()
        alpha = alpha_new.mean(dim=2)

        # MSE and convergence
        III = x_hat_pre - x_hat_new
        NMSE_iter = torch.sum(torch.abs(III)**2) / torch.sum(torch.abs(x_hat_new)**2)
        MSE[t] = NMSE_iter

        if t > 15 and MSE[t] >= MSE[t - 1]:
            x_hat = x_hat_pre.clone()
            break

        x_hat = x_hat_new.clone()
        var_hat = var_hat_new.clone()
        V = V_new.clone()
        Z = Z_new.clone()
        t += 1

    est_k = x_hat[:, 0, 0].real.round().int()
    est_k = est_k.view(9, -1).sum(dim=0)
    return est_k
