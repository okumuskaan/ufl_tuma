import torch

# -------------------------------------------------------------
# Log complex Gaussian likelihood
# -------------------------------------------------------------
def log_complex_gaussian_likelihood(r, all_Covs, T, nP, keepdims=False, device=None):
    """
    Torch-based version of the complex Gaussian log-likelihood.
    r: (..., F)
    all_Covs: (Kmax+1, Ns, F)
    T: (F,)
    """
    device = device or r.device
    denom = T + nP * all_Covs  # shape compatible for broadcasting
    log_likelihood = -((torch.abs(r)**2) / denom + torch.log(torch.pi * denom)).sum(dim=-1, keepdim=keepdims)
    return log_likelihood


# -------------------------------------------------------------
# Stable log-sum-exp (torch version)
# -------------------------------------------------------------
def log_sum_exp_trick(logai, i_axis, keepdims=True):
    maxlogai, _ = torch.max(logai, dim=i_axis, keepdim=True)
    res = maxlogai + torch.log(torch.sum(torch.exp(logai - maxlogai), dim=i_axis, keepdim=True))
    return res if keepdims else res.squeeze(i_axis)

def log_sum_exp_trick_foronsager(logai, i_axis, keepdims=True):
    maxlogai, _ = torch.max(logai.real, dim=i_axis, keepdim=True) # ensure real for max operation, delete later if necessary
    res = maxlogai + torch.log(torch.sum(torch.exp(logai - maxlogai), dim=i_axis, keepdim=True))
    return res if keepdims else res.squeeze(i_axis)


# -------------------------------------------------------------
# Compute log-posteriors
# -------------------------------------------------------------
def compute_logposteriors_with_logsumexptrick(y, all_Covs, T, nP, Ns, log_priors=0.0, device=None):
    device = device or y.device
    logai = -((torch.abs(y)**2) / (T + nP * all_Covs) + torch.log(torch.pi * (T + nP * all_Covs))).sum(dim=-1) - torch.log(torch.tensor(Ns, device=device, dtype=torch.float64))

    maxlogai, _ = torch.max(logai, dim=-1, keepdim=True)
    logposteriors = (maxlogai[:, 0] + torch.log(torch.sum(torch.exp(logai - maxlogai), dim=-1))) + log_priors
    est_type = torch.argmax(logposteriors).item()

    return est_type, logposteriors


# -------------------------------------------------------------
# Normalize posteriors
# -------------------------------------------------------------
def normalize_posteriors(logposteriors):
    max_post = torch.max(logposteriors)
    log_sum = max_post + torch.log(torch.sum(torch.exp(logposteriors - max_post)))
    return torch.exp(logposteriors - log_sum)


# -------------------------------------------------------------
# Numerically stable log(1 - exp(x))
# -------------------------------------------------------------
def safe_log1mexp(x, epsilon=1e-12):
    x_clipped = torch.clamp(x, max=0.0)
    return torch.log(1 - torch.exp(x_clipped) + epsilon)

def safe_log1mexp_for_onsager(x, epsilon=1e-12):
    # Separate real and imaginary parts
    x_real = torch.clamp(x.real, max=0.0)
    # Recombine with the original imaginary part (phase preserved)
    x_safe = torch.complex(x_real, x.imag)
    return torch.log(1 - torch.exp(x_safe) + epsilon)


# -------------------------------------------------------------
# Bayesian multiplicity estimation (no Onsager)
# -------------------------------------------------------------
def bayesian_channel_multiplicity_estimation_sampbasedapprox(r, all_covs, taus, nP, log_priors=0.0, device=None):
    device = device or r.device
    loglikelihoods_ai = log_complex_gaussian_likelihood(r, all_covs, taus, nP, keepdims=True, device=device)

    log_denum_ai = loglikelihoods_ai[1:]
    maxlog_denum_ai, _ = torch.max(log_denum_ai, dim=-2, keepdim=True)
    logdenum = maxlog_denum_ai + torch.log(torch.sum(torch.exp(log_denum_ai - maxlog_denum_ai), dim=-2, keepdim=True))

    log_num_ai = torch.log(torch.sqrt(torch.tensor(nP, device=device)) * all_covs[1:]) - torch.log(taus + nP * all_covs[1:]) + log_denum_ai
    maxlog_num_ai, _ = torch.max(log_num_ai, dim=-2, keepdim=True)
    lognum = maxlog_num_ai + torch.log(torch.sum(torch.exp(log_num_ai - maxlog_num_ai), dim=-2, keepdim=True))

    log_leftpart = lognum - logdenum

    log_posteriors_ai = loglikelihoods_ai + log_priors.unsqueeze(-1).unsqueeze(-1)
    maxlog_posteriors_ai, _ = torch.max(log_posteriors_ai, dim=-2, keepdim=True)
    log_posteriors = maxlog_posteriors_ai + torch.log(torch.sum(torch.exp(log_posteriors_ai - maxlog_posteriors_ai), dim=-2, keepdim=True))

    max_post = torch.max(log_posteriors)
    log_sum = max_post + torch.log(torch.sum(torch.exp(log_posteriors - max_post)))
    log_rightpart = log_posteriors - log_sum

    est_X = r * torch.sum(torch.exp(log_rightpart[1:, 0]) * torch.exp(log_leftpart[:, 0]), dim=0)
    return est_X

def bayesian_channel_multiplicity_estimation_sampbasedapprox_vectorized(R_u, all_covs, taus, nP, log_priors, device=None):
    device = device or R_u.device
    M, F = R_u.shape
    loglikelihoods_ai_allm = log_complex_gaussian_likelihood(R_u.view(M,1,1,F), all_covs, taus, nP, keepdims=True, device=device)

    log_denum_ai_allm = loglikelihoods_ai_allm[:,1:]
    maxlog_denum_ai_allm, _ = torch.max(log_denum_ai_allm, dim=-2, keepdim=True)
    logdenum_allm = maxlog_denum_ai_allm + torch.log(torch.sum(torch.exp(log_denum_ai_allm - maxlog_denum_ai_allm), dim=-2, keepdim=True))

    log_num_ai_allm = torch.log(torch.sqrt(torch.tensor(nP, device=device, dtype=all_covs.dtype)) * all_covs[1:]) - torch.log(taus + nP * all_covs[1:]) + log_denum_ai_allm
    maxlog_num_ai_allm, _ = torch.max(log_num_ai_allm, dim=-2, keepdim=True)
    lognum_allm = maxlog_num_ai_allm + torch.log(torch.sum(torch.exp(log_num_ai_allm - maxlog_num_ai_allm), dim=-2, keepdim=True))
    
    log_leftpart_allm = lognum_allm - logdenum_allm
    
    log_priors = log_priors.to(dtype=all_covs.real.dtype, device=device)
    log_posteriors_ai_allm = loglikelihoods_ai_allm + log_priors.unsqueeze(-1).unsqueeze(-1)
    maxlog_posteriors_ai_allm, _ = torch.max(log_posteriors_ai_allm, dim=-2, keepdim=True)
    log_posteriors_allm = maxlog_posteriors_ai_allm + torch.log(torch.sum(torch.exp(log_posteriors_ai_allm - maxlog_posteriors_ai_allm), dim=-2, keepdim=True))
    
    max_post_allm = torch.max(log_posteriors_allm, dim=1, keepdims=True)[0]
    log_sum_allm = max_post_allm + torch.log(torch.exp(log_posteriors_allm - max_post_allm).sum(axis=1, keepdims=True))
    log_rightpart_allm = log_posteriors_allm - log_sum_allm
    
    est_X_allm = R_u * torch.sum(torch.exp(log_rightpart_allm[:,1:,0]) * torch.exp(log_leftpart_allm[:,:,0]), dim=1)
    return est_X_allm


# -------------------------------------------------------------
# Bayesian estimation with Onsager correction
# -------------------------------------------------------------
def bayesian_channel_multiplicity_estimation_sampbasedapprox_with_onsager(r, all_covs, taus, nP, log_priors=0.0, device=None):
    device = device or r.device
    loglikelihoods_ai = log_complex_gaussian_likelihood(r, all_covs, taus, nP, keepdims=True, device=device)

    log_denum_ai = loglikelihoods_ai[1:]
    logdenum = log_sum_exp_trick_foronsager(log_denum_ai, i_axis=-2)

    log_num_ai = torch.log(torch.sqrt(torch.tensor(nP, device=device)) * all_covs[1:]) - torch.log(taus + nP * all_covs[1:]) + log_denum_ai
    lognum = log_sum_exp_trick_foronsager(log_num_ai, i_axis=-2)
    log_F_brk = lognum - logdenum

    log_posteriors_ai = loglikelihoods_ai + log_priors.unsqueeze(-1).unsqueeze(-1)
    log_posteriors = log_sum_exp_trick_foronsager(log_posteriors_ai, i_axis=-2)
    log_sumpost = log_sum_exp_trick_foronsager(log_posteriors, i_axis=0).reshape(-1)[0]
    log_rightpart = log_posteriors - log_sumpost
    log_G_rk = log_posteriors[1:] - log_sumpost

    H = torch.sum(torch.exp(log_rightpart[1:, 0]) * torch.exp(log_F_brk[:, 0]), dim=0)
    est_x = r * H
    est_k = torch.argmax(log_posteriors).item()

    log_estB_rki = loglikelihoods_ai.unsqueeze(-1) + torch.log(-r.conj().unsqueeze(-1)) - torch.log((taus + nP * all_covs).unsqueeze(-1))
    log_estB_rk = log_sum_exp_trick_foronsager(log_estB_rki, i_axis=1, keepdims=False)

    log_estA_brk = log_sum_exp_trick_foronsager(
        torch.log(torch.sqrt(torch.tensor(nP, device=device)) * all_covs[1:]).unsqueeze(-2)
        - torch.log(taus + nP * all_covs[1:]).unsqueeze(-2)
        + log_estB_rki[1:], i_axis=1, keepdims=False
    )

    log_estF_brk = log_estA_brk + safe_log1mexp_for_onsager(log_F_brk + log_estB_rk[1:] - log_estA_brk) - logdenum
    log_estC_rk = log_estB_rk + log_priors.unsqueeze(-1).unsqueeze(-1)
    log_estD_r = log_sum_exp_trick_foronsager(log_estC_rk, i_axis=0)
    log_estG_rk = log_estC_rk[1:] + safe_log1mexp_for_onsager(log_G_rk + log_estD_r - log_estC_rk[1:]) - log_sumpost

    est_H_b = torch.sum(torch.exp(log_F_brk + log_estG_rk) + torch.exp(log_estF_brk + log_G_rk), dim=0)
    Qum = torch.diag(H) + r * est_H_b

    return est_x, est_k, Qum


# -------------------------------------------------------------
# Type estimation (sampling-based, log-sum-exp)
# -------------------------------------------------------------
def estimate_type_samplingbased_logsumexp(R, T, all_Covs, M, U, nP, log_priors, device=None):
    device = device or R[0].device
    Mus = [M] * U
    est_k = torch.zeros(sum(Mus), dtype=torch.int64, device=device)
    posteriors = []

    m = 0
    Kmax = all_Covs.shape[1]
    Ns = all_Covs.shape[-2]

    for u, Mu in enumerate(Mus):
        posteriors_u = torch.zeros((Mu, Kmax), dtype=torch.float64, device=device)
        for mu in range(M):
            est_mult, logpost = compute_logposteriors_with_logsumexptrick(
                R[u][mu], all_Covs[u], T, nP, Ns, log_priors[u][mu], device=device
            )
            posteriors_u[mu] = normalize_posteriors(logpost)
            est_k[m] = est_mult
            m += 1
        posteriors.append(posteriors_u)

    return est_k, posteriors

def estimate_type_samplingbased_logsumexp_vectorized(R, T, all_Covs, M, nP, log_priors, device=None):
    device = device or R[0].device
    U = len(R)
    est_k = torch.zeros(M*U, dtype=torch.int64, device=device)
    Ns = all_Covs.shape[-2]
    F = all_Covs.shape[-1]
    for u in range(U):
        logai_allm = -((torch.abs(R[u].view(M,1,1,F))**2) / (T + nP * all_Covs[u]) + torch.log(torch.pi * (T + nP * all_Covs[u]))).sum(dim=-1) - torch.log(torch.tensor(Ns, device=device, dtype=torch.float64))
        maxlogai_allm, _ = torch.max(logai_allm, dim=-1, keepdim=True)
        logposteriors_allm = (maxlogai_allm[:,:,0] + torch.log(torch.sum(torch.exp(logai_allm - maxlogai_allm), dim=-1))) + log_priors[u][0]
        est_type = torch.argmax(logposteriors_allm,dim=-1)
        est_k[u*M:(u+1)*M] = est_type
    return est_k

def estimate_type_samplingbased_logsumexp_vectorized_more(R, T, all_Covs, M, nP, log_priors, device=None):
    U, Kmax, Ns, F = all_Covs.shape
    temp = (T + nP * all_Covs).view(U,1,Kmax,Ns,F)
    logai_allum = -((torch.abs(R.view(U,M,1,1,F))**2) / (temp) + torch.log(torch.pi * temp)).sum(dim=-1) - torch.log(torch.tensor(Ns, device=device, dtype=torch.float64))
    maxlogai_allum, _ = torch.max(logai_allum, dim=-1, keepdim=True)
    logposteriors_allum = (maxlogai_allum[:,:,:,0] + torch.log(torch.sum(torch.exp(logai_allum - maxlogai_allum), dim=-1))) + log_priors[0][0]
    return torch.argmax(logposteriors_allum,dim=-1).reshape(-1)
