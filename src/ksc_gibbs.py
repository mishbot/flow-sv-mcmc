import time
from pathlib import Path

import numpy as np
from scipy.stats import uniform, norm, beta, invgamma, truncnorm
from scipy.special import softmax
from numba import njit
from tqdm import tqdm


def _sample_s(z, h, q, m, v2, rng):
    """Categorical posterior P(s_t = k | h_t, z_t) ∝ q_k * N(z_t | h_t + m_k, v_k^2).
    Returns int array shape (T,) with values in {0, ..., 6}."""
    log_w = np.log(q)[None, :] + norm.logpdf(z[:, None],                            # (T, 7)
                                             loc=h[:, None] + m[None, :],
                                             scale=np.sqrt(v2)[None, :])
    w = softmax(log_w, axis=1)

    # Using the PDF above, create corresponding CDF and sample 7 indicators
    w_cdf = np.cumsum(w, axis=1)
    u = uniform.rvs(size=z.shape[0], random_state=rng)[:, None]
    s = np.sum(w_cdf < u, axis=1)
    return np.clip(s, 0, 6).astype(int)


# numba speedup is ~30x
@njit(cache=True)
def _ffbs(z_tilde, obs_var, mu, phi, sigma2_eta, eps, T):
    """Kalman forward filter + Carter-Kohn backward sample. Pre-generated eps
    carries randomness in (numba can't see np.random.Generator)."""
    # Prediction states
    a_p = np.empty(T)
    P_p = np.empty(T)

    # Filter states
    a_f = np.empty(T)
    P_f = np.empty(T)

    # Initialization is at AR(1)
    a_cur = mu
    P_cur = sigma2_eta / (1.0 - phi * phi)

    # Forward filter
    for t in range(T):
        if t > 0:
            # Forward prediction step: state + observation eq's
            a_cur = mu + phi * (a_cur - mu)         # predict mean
            P_cur = sigma2_eta + phi * phi * P_cur  # predict variance

        a_p[t] = a_cur
        P_p[t] = P_cur

        # Forward update step
        K = P_cur / (P_cur + obs_var[t])       # Kalman gain
        a_cur = a_cur + K * (z_tilde[t] - a_cur)
        P_cur = (1.0 - K) * P_cur

        a_f[t] = a_cur
        P_f[t] = P_cur

    # Backward sampler. Why not RTS?
    # - Because RTS yields marginals, but h_t observations are jointly correlated.
    # - FFBS gives joint samples with correct correlations.
    h = np.empty(T)
    h[-1] = a_f[-1] + np.sqrt(P_f[-1]) * eps[T - 1]

    for t in range(T - 2, -1, -1):
        # Given the h_{t+1} we've just drawn, the conditional h_t | h_{t+1}, z_{1:T}
        # is Gaussian. The clean form of the variance is the one below.
        # Algebraically: P_{t|t} * sigma_eta^2 / P_{t+1|t}.
        Kb = phi * P_f[t] / P_p[t + 1]    # Backward Kalman gain
        mean = a_f[t] + Kb * (h[t + 1] - a_p[t + 1])
        var = P_f[t] * sigma2_eta / P_p[t + 1]
        h[t] = mean + np.sqrt(var) * eps[t]

    return h


def _sample_h(z, s_idx, mu, phi, sigma2_eta, m, v2, rng):
    T = len(z)                                  
    z_tilde = z - m[s_idx]
    obs_var = v2[s_idx]
    eps = norm.rvs(size=T, random_state=rng)    # numba can't compile norm.rvs - main reason for helper
    return _ffbs(z_tilde, obs_var, mu, phi, sigma2_eta, eps, T)
        

def _sample_mu(h, phi, sigma2_eta, prior_var, rng):
    """Conjugate Normal posterior for mu. Combines N(0, prior_var) prior with
    the stationary h_1 term and (T-1) AR(1) transitions. Returns scalar."""
    T = len(h)
    drift = 1.0 - phi
    stat_scale = 1.0 - phi**2

    # Proper prior with finite variance generalizes KSC (1998) eq. 7, which is the special case prior_var = inf
    tau_post = 1.0 / prior_var + ((T - 1) * drift**2 + stat_scale) / sigma2_eta
    m_post = (drift * np.sum(h[1:] - phi * h[:-1]) + stat_scale * h[0]) / (sigma2_eta * tau_post)
    return norm.rvs(loc=m_post, scale=np.sqrt(1.0 / tau_post), random_state=rng)


def _sample_sigma2_eta(h, mu, phi, shape, rate, rng):
    """Conjugate Inverse-Gamma posterior for sigma_eta^2 given (h, mu, phi).
    Returns scalar."""
    T = len(h)
    post_a = shape + T / 2.0
    post_b = rate + 0.5 * ((h[0] - mu)**2 * (1.0 - phi**2)
                           + np.sum((h[1:] - mu - phi * (h[:-1] - mu))**2))
    return invgamma.rvs(post_a, scale=post_b, random_state=rng)


def _sample_phi(h, mu, sigma2_eta, phi_curr, beta_a, beta_b, rng):
    """MH update for phi with a truncated-Gaussian OLS proposal. Acceptance
    factor is KSC eq. (6)'s g(phi): Beta prior + stationary h_1 marginal,
    the AR(1) transition sum cancels with the proposal density. Returns
    (phi_new, accepted)."""
    def g_mh(phi):
        stat_scale = 1 - phi**2
        prior = beta.logpdf((phi + 1) / 2, beta_a, beta_b)
        return (prior + 0.5 * np.log(stat_scale)
                - stat_scale * (h[0] - mu)**2 / (2 * sigma2_eta))

    S = np.sum((h[:-1] - mu)**2)
    phi_hat = np.sum((h[1:] - mu) * (h[:-1] - mu)) / S
    sd_phi = np.sqrt(sigma2_eta / S)

    # Truncated Gaussian ensures the proposed phi is in (-1, 1)
    a, b = (-1 - phi_hat) / sd_phi, (1 - phi_hat) / sd_phi
    phi_proposed = truncnorm.rvs(a, b, loc=phi_hat, scale=sd_phi, random_state=rng)

    # MH Acceptance
    mh_alpha = g_mh(phi_proposed) - g_mh(phi_curr)
    if np.log(uniform.rvs(random_state=rng)) < mh_alpha:
        return phi_proposed, True
    return phi_curr, False


def run_ksc_gibbs(y, n_iter, n_burn, prior_mu_var, prior_sigma2_eta_gamma, prior_phi_beta, q, m, v2, offset=1e-6, seed=42, verbose=True,
                  progress=None, progress_label=None, keep_h=True):
    """KSC 1998 mixture-of-normals Gibbs sampler for the SV model. Returns a
    canonical sample dict matching the run_flow_mcmc schema (keys: mu, phi,
    sigma_eta, h, accept_phi, n_warmup, n_draws, n_chains, n_dim, seed)."""
    rng = np.random.default_rng(seed)
    T = len(y)
    log_y2_c = np.log(y**2 + offset) + 1.2704

    mus = np.empty(n_iter)
    phis = np.empty(n_iter)
    sigma2_etas = np.empty(n_iter)
    hs = np.empty((n_iter, T)) if keep_h else None
    h_mids = np.empty(n_iter)
    h_lasts = np.empty(n_iter)
    accs = np.empty(n_iter, dtype=bool)
    mid = T // 2

    mu = log_y2_c.mean()
    phi = 0.9
    sigma2_eta = 0.1
    h = np.full(T, mu)

    s2_shape, s2_rate = prior_sigma2_eta_gamma
    phi_beta_a, phi_beta_b = prior_phi_beta

    iterator = tqdm(range(n_iter), desc="KSC Gibbs", disable=not verbose) if progress is None else range(n_iter)
    for i in iterator:
        s = _sample_s(log_y2_c, h, q, m, v2, rng)
        h = _sample_h(log_y2_c, s, mu, phi, sigma2_eta, m, v2, rng)
        mu = _sample_mu(h, phi, sigma2_eta, prior_mu_var, rng)
        sigma2_eta = _sample_sigma2_eta(h, mu, phi, s2_shape, s2_rate, rng)
        phi, acc = _sample_phi(h, mu, sigma2_eta, phi, phi_beta_a, phi_beta_b, rng)

        mus[i] = mu
        phis[i] = phi
        sigma2_etas[i] = sigma2_eta
        if keep_h:
            hs[i] = h
        h_mids[i] = h[mid]
        h_lasts[i] = h[-1]
        accs[i] = acc

        postfix = {
            "mu": f"{mu:+.3f}",
            "phi": f"{phi:.4f}",
            "sigma_eta": f"{np.sqrt(sigma2_eta):.3f}",
        }
        if progress_label is not None:
            postfix = {"chain": progress_label, "seed": int(seed), **postfix}

        if progress is None:
            iterator.set_postfix(**postfix, refresh=False)
        else:
            progress.update(1)
            if (i + 1) % 50 == 0 or i + 1 == n_iter:
                progress.set_postfix(**postfix, refresh=False)


    if verbose:
        print(f"\nUsing burn-in of {n_burn} iterations.")

    result = {
        "mu":         mus[n_burn:].astype(np.float32),
        "phi":        phis[n_burn:].astype(np.float32),
        "sigma_eta":  np.sqrt(sigma2_etas[n_burn:]).astype(np.float32),
        "accept_phi": float(accs[n_burn:].mean()),
        "n_warmup":   n_burn,
        "n_draws":    n_iter - n_burn,
        "n_chains":   1,
        "n_dim":      T + 3,
        "seed":       int(seed)
    }

    if keep_h:
        result["h"] = hs[n_burn:].astype(np.float32)
    else:
        result["h_mid"] = h_mids[n_burn:].astype(np.float32)
        result["h_last"] = h_lasts[n_burn:].astype(np.float32)

    return result


def _stack_chains(chains, key):
    """Stack a single-chain KSC output key into (n_chains, n_draws, ...)."""
    return np.stack([chain[key] for chain in chains])


def _summarize_h(h):
    """Return latent-path summaries needed for diagnostics and comparison."""
    h_flat = h.reshape(-1, h.shape[-1])
    return {
        "h_mid": h[:, :, h.shape[-1] // 2].astype(np.float32),
        "h_last": h[:, :, -1].astype(np.float32),
        "h_mean": h_flat.mean(axis=0).astype(np.float32),
        "h_ci": np.quantile(h_flat, [0.025, 0.975], axis=0).astype(np.float32),
    }


def run_ksc_chains(y, n_chains, n_iter, n_burn, prior_mu_var, prior_sigma2_eta_gamma, prior_phi_beta, 
                   q, m, v2, offset=1e-6, seed=42, seed_stride=1, verbose=True, keep_h=False, summarize_h_path=True):
    """Run multiple independent KSC chains and return stacked samples.

    Set summarize_h_path=False for real-data runs when only h_mid and h_last
    are needed. This avoids stacking the full latent path and computing a
    pointwise credible interval over all draws.
    """
    y = np.asarray(y)
    chains = []
    chain_elapsed = []
    chain_seeds = np.array([int(seed + seed_stride * c) for c in range(n_chains)])
    keep_h = bool(keep_h or summarize_h_path)

    progress = tqdm(total=n_chains * n_iter, desc="KSC Gibbs", disable=not verbose)
    try:
        for c, chain_seed in enumerate(chain_seeds):
            t0 = time.perf_counter()
            chain = run_ksc_gibbs(
                y,
                n_iter=n_iter,
                n_burn=n_burn,
                prior_mu_var=prior_mu_var,
                prior_sigma2_eta_gamma=prior_sigma2_eta_gamma,
                prior_phi_beta=prior_phi_beta,
                q=q,
                m=m,
                v2=v2,
                offset=offset,
                seed=int(chain_seed),
                verbose=False,
                progress=progress if verbose else None,
                progress_label=f"{c + 1}/{n_chains}",
                keep_h=keep_h,
            )
            chain_elapsed.append(time.perf_counter() - t0)
            chains.append(chain)
    finally:
        progress.close()

    mu = _stack_chains(chains, "mu").astype(np.float32)
    phi = _stack_chains(chains, "phi").astype(np.float32)
    sigma_eta = _stack_chains(chains, "sigma_eta").astype(np.float32)

    result = {
        "mu": mu,
        "phi": phi,
        "sigma_eta": sigma_eta,
        "accept_phi": np.array([chain["accept_phi"] for chain in chains], dtype=np.float32),
        "elapsed_sec": float(np.sum(chain_elapsed)),
        "chain_elapsed_sec": np.asarray(chain_elapsed, dtype=np.float32),
        "n_warmup": int(chains[0]["n_warmup"]),
        "n_draws": int(mu.shape[1]),
        "n_chains": int(n_chains),
        "n_dim": int(y.shape[0] + 3),
        "seed": int(seed),
        "chain_seeds": chain_seeds.astype(np.int64)
    }

    if keep_h:
        h = _stack_chains(chains, "h").astype(np.float32)
        result["h_mid"] = h[:, :, h.shape[-1] // 2].astype(np.float32)
        result["h_last"] = h[:, :, -1].astype(np.float32)
        if summarize_h_path:
            result.update({
                key: value
                for key, value in _summarize_h(h).items()
                if key not in ("h_mid", "h_last")
            })
        if keep_h:
            result["h"] = h
    else:
        result["h_mid"] = _stack_chains(chains, "h_mid").astype(np.float32)
        result["h_last"] = _stack_chains(chains, "h_last").astype(np.float32)

    return result


def save_ksc_result(path, result):
    """Save a run_ksc_chains result to a compressed .npz file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **result)
