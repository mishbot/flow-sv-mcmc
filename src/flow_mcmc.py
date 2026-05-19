import time

import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy.stats import norm, beta, gamma

from flowMC.Sampler import Sampler
from flowMC.resource_strategy_bundle.RQSpline_GRW import RQSpline_GRW_Bundle
from flowMC.resource_strategy_bundle.RQSpline_MALA import RQSpline_MALA_Bundle

import modal

from src.config import (SEED,
                        PRIOR_MU_VAR, PRIOR_PHI_BETA, PRIOR_SIGMA_ETA_GAMMA,
                        FLOW_N_HIDDEN, FLOW_N_LOCAL, FLOW_N_GLOBAL, FLOW_N_TRAINING, FLOW_N_PRODUCTION, 
                        FLOW_THETA0_SCALE, FLOW_GRW_STEPSIZE, FLOW_N_EPOCHS, FLOW_N_CHAINS, FLOW_INIT_THETA,
                        FLOW_MAX_STORED_DRAWS, FLOW_LOCAL_KERNEL, FLOW_MALA_STEPSIZE, FLOW_LOCAL_THINNING)

def _log_prior(mu, phi, sigma_eta, prior_mu_var, beta_a, beta_b, gamma_a, gamma_b):
    """Evaluate the SV parameter prior density on the constrained scale."""
    log_p_mu = norm.logpdf(mu, loc=0, scale=jnp.sqrt(prior_mu_var))
    
    log_p_phi = beta.logpdf((phi + 1) / 2, a=beta_a, b=beta_b)
    
    sigma2_eta = sigma_eta ** 2
    log_p_s2inv = gamma.logpdf(1.0 / sigma2_eta, a=gamma_a, scale=1.0 / gamma_b)            # jax.scipy.stats lacks IG
    # Two Jacobians: gamma.logpdf returns log p(s2inv), but we want log p(s)
    # Chain rule: s2inv -> s2 adds -2log(s^2), and s^2 -> s adds log(2s).
    log_p_sigma_eta = log_p_s2inv + jnp.log(2.0 * sigma_eta) - 2.0 * jnp.log(sigma2_eta)
    
    return log_p_mu + log_p_phi + log_p_sigma_eta


def _h_from_z(z, mu, phi, sigma_eta):
    """Reconstruct the latent log-volatility path from non-centered innovations."""
    h0 = mu + sigma_eta / jnp.sqrt(1.0 - phi**2) * z[0]

    def step(h_prev, z_t):
        h_t = mu + phi * (h_prev - mu) + sigma_eta * z_t
        return h_t, h_t

    _, h_tail = jax.lax.scan(step, h0, z[1:])
    return jnp.concatenate([h0[None], h_tail])


@jax.jit
def _log_post(state, data):
    """Evaluate the non-centered flowMC target density on the unconstrained scale."""
    prior_mu_var = data['prior_mu_var']
    beta_a, beta_b = data['prior_phi_beta']
    gamma_a, gamma_b = data['prior_sigma_eta_gamma']
    
    mu = state[0]
    phi = jnp.tanh(state[1])    # reals -> (-1, 1), differentiable for flow
    sigma_eta = jnp.exp(state[2])
    z = state[3:]
    
    # Non-centered latent path (similar to NUTS implementation) - need to reconstruct h
    h = _h_from_z(z, mu, phi, sigma_eta)
    
    # Collect log probs, output
    log_prior = _log_prior(mu, phi, sigma_eta, prior_mu_var, beta_a, beta_b, gamma_a, gamma_b)
    log_p_z = norm.logpdf(z, loc=0.0, scale=1.0).sum()
    log_p_y = norm.logpdf(data["y"], loc=0.0, scale=jnp.exp(h / 2)).sum()
    log_jac = jnp.log(1.0 - phi**2) + state[2]  # Jacobians: log(1-phi^2) from tanh, x[2]=log(sigma_eta) from exp
    return log_prior + log_p_z + log_p_y + log_jac


def run_flow_mcmc(y_obs, n_hidden=FLOW_N_HIDDEN, n_local=FLOW_N_LOCAL, n_global=FLOW_N_GLOBAL, n_training=FLOW_N_TRAINING, 
                  n_production=FLOW_N_PRODUCTION, n_epochs=FLOW_N_EPOCHS, n_chains=FLOW_N_CHAINS, prior_mu_var=PRIOR_MU_VAR, 
                  prior_phi_beta=PRIOR_PHI_BETA, prior_sigma_eta_gamma=PRIOR_SIGMA_ETA_GAMMA, theta_init=FLOW_INIT_THETA, 
                  local_kernel=FLOW_LOCAL_KERNEL, grw_step_size=FLOW_GRW_STEPSIZE, mala_step_size=FLOW_MALA_STEPSIZE,
                  adapt_step_size=True, local_thinning=FLOW_LOCAL_THINNING, theta_init_scale=FLOW_THETA0_SCALE,
                  max_stored_draws=FLOW_MAX_STORED_DRAWS, seed=SEED, verbose=True):
    """Run flowMC for the SV model using either a GRW or MALA local kernel."""
    if len(theta_init) != 3:
        raise ValueError("theta_init must contain (mu, phi, sigma_eta). Use None for mu to initialize from the data.")
    mu_init, phi_init, sigma_eta_init = theta_init
    if phi_init is None or sigma_eta_init is None:
        raise ValueError("theta_init must provide finite phi and sigma_eta initial values.")
    if not (-1.0 < float(phi_init) < 1.0):
        raise ValueError("theta_init phi must be strictly between -1 and 1.")
    if float(sigma_eta_init) <= 0.0:
        raise ValueError("theta_init sigma_eta must be positive.")

    y_obs = jnp.asarray(y_obs)
    T = len(y_obs)
    if T == 0:
        raise ValueError("y_obs must contain at least one observation.")
    n_dim = T + 3       # T non-centered z innovations + unconstrained theta
    
    k_bundle, k_sampler, k_theta_noise, k_z = jax.random.split(jax.random.PRNGKey(seed), 4)
    
    # Initial position: method-of-moments + small amount of noise across chains
    mu_mom = jnp.mean(jnp.log(y_obs**2 + 1e-6)) + 1.2704
    mu_init = mu_mom if mu_init is None else jnp.asarray(mu_init)
    
    # Theta baseline - same for all chains with noise added below
    theta_center = jnp.array([mu_init, jnp.arctanh(phi_init), jnp.log(sigma_eta_init)])
    
    # Add per-chain noise to theta. Latent diversity comes from iid z innovations
    theta_noise_scale = jnp.asarray(theta_init_scale)
    if theta_noise_scale.shape != (3,):
        raise ValueError("theta_init_scale must contain three values for (mu, atanh(phi), log(sigma_eta)).")
    theta_per_chain = theta_center + theta_noise_scale * jax.random.normal(k_theta_noise, (n_chains, 3))
    
    # Simulate non-centered path
    z_init = jax.random.normal(k_z, (n_chains, T))    
    
    # Concatenate per-chain position
    init_pos = jnp.concatenate([theta_per_chain, z_init], axis=1)
    
    local_kernel = local_kernel.lower()
    
    # Bundle for flowMC>=5.0.0 sampler
    bundle_kwargs = dict(
        rng_key=k_bundle,
        n_chains=n_chains,
        n_dims=n_dim,
        logpdf=_log_post,
        n_local_steps=n_local,
        n_global_steps=n_global,
        n_training_loops=n_training,
        n_production_loops=n_production,
        n_epochs=n_epochs,
        rq_spline_hidden_units=[n_hidden, n_hidden],
        adapt_step_size=adapt_step_size,
        local_thinning=local_thinning,
        verbose=verbose,
    )

    if local_kernel == "grw":
        if len(grw_step_size) != 4:
            raise ValueError("grw_step_size must contain four values for (mu, atanh(phi), log(sigma_eta), z).")
        step_size = jnp.concatenate([
            jnp.array(grw_step_size[:3]),     # mu, atanh(phi), log(sigma_eta)
            jnp.full(T, grw_step_size[3])     # z_t non-centered innovations
        ])
        bundle = RQSpline_GRW_Bundle(**bundle_kwargs, grw_step_size=step_size)
    elif local_kernel == "mala":
        if len(mala_step_size) != 4:
            raise ValueError("mala_step_size must contain four values for (mu, atanh(phi), log(sigma_eta), z).")
        step_size = jnp.concatenate([
            jnp.array(mala_step_size[:3]),    # mu, atanh(phi), log(sigma_eta)
            jnp.full(T, mala_step_size[3])    # z_t non-centered innovations
        ])
        bundle = RQSpline_MALA_Bundle(**bundle_kwargs, mala_step_size=step_size)
    else:
        raise ValueError("local_kernel must be 'grw' or 'mala'")
    
    sampler = Sampler(
        n_dim=n_dim, 
        n_chains=n_chains,
        rng_key=k_sampler,
        resource_strategy_bundles=bundle
    )
    
    init_data = {
        'y': y_obs,
        'prior_mu_var': prior_mu_var,
        'prior_phi_beta': jnp.asarray(prior_phi_beta),
        'prior_sigma_eta_gamma': jnp.asarray(prior_sigma_eta_gamma),
    }
    
    t0 = time.perf_counter()
    sampler.sample(init_pos, data=init_data)
    elapsed_sec = time.perf_counter() - t0
    
    pos_prod = np.asarray(sampler.resources["positions_production"].data)  # (n_chains, n_steps, n_dim)
    n_raw_draws = pos_prod.shape[1]

    # flowMC stores every local production state. At T=2500 this is too large
    # to reconstruct into a full latent h tensor, so keep evenly spaced draws.
    if max_stored_draws is not None and n_raw_draws > max_stored_draws:
        draw_idx = np.linspace(0, n_raw_draws - 1, max_stored_draws, dtype=np.int64)
        pos_prod = pos_prod[:, draw_idx, :]
        draw_thin = int(np.median(np.diff(draw_idx))) if draw_idx.size > 1 else n_raw_draws
    else:
        draw_idx = None
        draw_thin = 1
    
    # Reconstruct h (later) from z
    mu_chain = pos_prod[:, :, 0].astype(np.float32)
    phi_chain = np.tanh(pos_prod[:, :, 1]).astype(np.float32)
    sigma_eta_chain = np.exp(pos_prod[:, :, 2]).astype(np.float32)
    z_chain = pos_prod[:, :, 3:].astype(np.float32)
    
    h_chain = np.asarray(
        jax.vmap(
            jax.vmap(_h_from_z, in_axes=(0, 0, 0, 0)),
            in_axes=(0, 0, 0, 0),
        )(
            jnp.asarray(z_chain),
            jnp.asarray(mu_chain),
            jnp.asarray(phi_chain),
            jnp.asarray(sigma_eta_chain),
        )
    ).astype(np.float32)
    
    # Only need this for comparison, h and z are too large to store
    h_mean = h_chain.mean(axis=(0, 1))
    h_ci = np.quantile(h_chain, [0.025, 0.975], axis=(0, 1))
    h_mid_chain = h_chain[:, :, T // 2]
    h_last_chain = h_chain[:, :, -1]
    del h_chain
    
    # Process diagnostics
    local_accept = np.asarray(sampler.resources["local_accs_production"].data)
    global_accept = np.asarray(sampler.resources["global_accs_production"].data)
    log_prob = np.asarray(sampler.resources["log_prob_production"].data)

    if draw_idx is not None:
        if local_accept.ndim >= 2 and local_accept.shape[1] == n_raw_draws:
            local_accept = local_accept[:, draw_idx]
        if global_accept.ndim >= 2 and global_accept.shape[1] == n_raw_draws:
            global_accept = global_accept[:, draw_idx]
        if log_prob.ndim >= 2 and log_prob.shape[1] == n_raw_draws:
            log_prob = log_prob[:, draw_idx]

    local_accept = np.where(np.isfinite(local_accept), local_accept, np.nan).astype(np.float32)
    global_accept = np.where(np.isfinite(global_accept), global_accept, np.nan).astype(np.float32)
    
    return {
        "mu": mu_chain.astype(np.float32),
        "phi": phi_chain.astype(np.float32),
        "sigma_eta": sigma_eta_chain.astype(np.float32),
        
        "h_mean": h_mean.astype(np.float32),
        "h_ci": h_ci.astype(np.float32),
        "h_mid": h_mid_chain.astype(np.float32),
        "h_last": h_last_chain.astype(np.float32),
        
        "log_prob": log_prob.astype(np.float32),
        "local_accept_prod": local_accept,
        "global_accept_prod": global_accept,
        "loss": np.asarray(sampler.resources["loss_buffer"].data),
        
        "n_chains": n_chains,
        "n_dim": n_dim,
        "n_draws": pos_prod.shape[1],
        "n_raw_draws": n_raw_draws,
        "draw_thin": draw_thin,
        "elapsed_sec": float(elapsed_sec),
        "local_kernel": local_kernel,
        "local_thinning": local_thinning,
        "seed": seed,
    }
