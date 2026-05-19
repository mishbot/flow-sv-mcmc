# Bayesian Calibration of Stochastic Volatility Models via Flow-Augmented MCMC

Mihail Dimitrov

This project implements and compares three Bayesian samplers for the stochastic volatility (SV) model used in financial time-series modeling:

1. **Kim-Shephard-Chib (KSC) mixture-of-normals Gibbs**: a specialized sampler for the SV model and the main reference posterior in this project.
2. **NUTS via PyMC**: a strong general-purpose Bayesian baseline using a non-centered latent representation.
3. **flowMC**: a flow-augmented MCMC approach that combines local MCMC moves with a learned rational-quadratic spline proposal.

All three samplers are run on a simulated SV series with known truth, S&P 500 returns, and AAPL returns. The final comparison looks at posterior recovery, mixing efficiency, posterior agreement, and downstream Value-at-Risk behavior.

The main notebook is [main.ipynb](main.ipynb).

## Project Motivation

The stochastic volatility model is a canonical Bayesian problem in quantitative finance. Daily returns are observed, but the underlying log-volatility is latent and persistent. The model used throughout the project is

$$
h_t = \mu + \phi(h_{t-1} - \mu) + \sigma_\eta \eta_t,
\qquad
\eta_t \sim N(0,1),
$$

$$
y_t = \exp(h_t / 2)\epsilon_t,
\qquad
\epsilon_t \sim N(0,1).
$$

Here, $h_t$ is the latent log-volatility and $y_t$ is the observed demeaned log-return. The parameter $\mu$ controls long-run volatility, $\phi$ controls persistence, and $\sigma_\eta$ controls the size of volatility shocks.

This posterior is difficult because the observation equation is not linear-Gaussian in the latent path, and financial volatility is highly persistent. When $\phi$ is close to one, adjacent latent states are strongly correlated, so naive one-at-a-time latent updates mix poorly.

## Data and Simulation

The notebook uses two real return series downloaded from `yfinance`:

- S&P 500 index returns, ticker `^GSPC`
- Apple returns, ticker `AAPL`

The data window is configured in [src/config.py](src/config.py). Returns are computed as log returns and then demeaned.

Before fitting real data, the notebook generates a simulated truth set from the same SV model. The simulated experiment uses

| Parameter | Meaning | Truth |
|---|---:|---:|
| $\mu$ | Long-run log-volatility mean | `-9.0` |
| $\phi$ | Volatility persistence | `0.97` |
| $\sigma_\eta$ | Volatility shock scale | `0.15` |

The simulated latent path is kept only for validation. In real data, the latent volatility path is never observed.

## Methods

### KSC Gibbs

The KSC sampler uses the Kim, Shephard, and Chib mixture approximation. Squaring returns and taking logs gives

$$
z_t = \log(y_t^2 + c) = h_t + \log(\epsilon_t^2).
$$

The remaining non-Gaussian error term $\log(\epsilon_t^2)$ is approximated by a seven-component Gaussian mixture. Conditional on the mixture indicators, the model becomes linear-Gaussian, so the full latent volatility path can be sampled jointly with forward-filter backward-sample (FFBS).

Each Gibbs sweep updates:

| Block | Quantity | Method |
|---|---|---|
| 1 | $s_{1:T}$ | Categorical mixture-state update |
| 2 | $h_{1:T}$ | FFBS joint latent-path draw |
| 3 | $\mu$ | Gaussian conditional update |
| 4 | $\sigma_\eta^2$ | Inverse-Gamma conditional update |
| 5 | $\phi$ | Metropolis-Hastings update |

The implementation lives in [src/ksc_gibbs.py](src/ksc_gibbs.py).

### NUTS via PyMC

NUTS is used as the main general-purpose Bayesian baseline. The PyMC model uses the same priors as the other samplers, but samples a non-centered latent innovation representation:

$$
z_t \sim N(0,1),
$$

then reconstructs

$$
h_t = \mu + \phi(h_{t-1} - \mu) + \sigma_\eta z_t.
$$

This helps reduce the strong posterior dependence between parameters and latent states. The sampler still has to explore a high-dimensional latent-state posterior, but it gives reliable convergence diagnostics such as R-hat, ESS, divergences, and trace plots.

### flowMC

The flowMC implementation uses the same non-centered target as the NUTS model, but moves through the posterior differently. It alternates between:

1. local MCMC moves,
2. global proposals from a trained rational-quadratic spline flow,
3. flow training on recent chain states.

flowMC operates on an unconstrained state vector

$$
x = (\mu, \phi_{\text{raw}}, \ell_\sigma, z_1,\ldots,z_T),
$$

with transformations

$$
\phi = \tanh(\phi_{\text{raw}}),
\qquad
\sigma_\eta = \exp(\ell_\sigma).
$$

The log posterior includes the corresponding Jacobian correction. The implementation lives in [src/flow_mcmc.py](src/flow_mcmc.py), and the remote execution helper lives in [src/pymc_modal.py](src/pymc_modal.py).

The main empirical finding is that the local component can produce usable draws after tuning, but the learned global proposal is hard to make effective in this high-dimensional latent-state posterior. In the project runs, the global flow acceptance was near zero, so flowMC mostly behaved like a tuned local sampler rather than a fully flow-accelerated sampler.

## Comparison Metrics

The final section compares samplers along three axes.

### Mixing Efficiency

Mixing is measured using bulk effective sample size per second. This is computed for

- $\mu$
- $\phi$
- $\sigma_\eta$
- a representative latent volatility state, $h_{T/2}$

Higher ESS/sec means the sampler produces more useful posterior draws per second. R-hat is also reported because ESS/sec is not meaningful if the chains have not mixed.

### Posterior Agreement

KSC is used as the reference posterior because it is the specialized sampler for this model. NUTS and flowMC are compared to KSC using marginal Wasserstein distances. Smaller distances indicate closer posterior agreement.

### Value-at-Risk Backtest

The downstream task is posterior predictive Value-at-Risk. For each sampler, the notebook simulates posterior predictive future returns and estimates one-day and ten-day VaR. The resulting forecasts are compared with realized losses using the Kupiec unconditional coverage test.

For GSPC and AAPL, this is a realized-loss backtest. For the simulated data, it is only a calibration check unless a separate held-out simulated series is generated.

The comparison utilities live in [src/comparison.py](src/comparison.py).

## Repository Layout

```text
.
|-- main.ipynb              # Main report notebook
|-- requirements.txt        # Python dependencies
|-- src/
|   |-- config.py           # Constants, priors, sampler settings
|   |-- diagnostics.py      # Diagnostics and plotting helpers
|   |-- ksc_gibbs.py        # KSC Gibbs sampler
|   |-- flow_mcmc.py        # flowMC implementation
|   |-- pymc_modal.py       # Modal helper code
|   `-- comparison.py       # ESS/sec, Wasserstein, VaR, Kupiec utilities
|-- tests/                  # Unit and smoke tests
|-- samples/                # Saved sampler outputs
|-- references/             # Papers used in the project
`-- outputs/                # Generated figures/results, if present
```

## Running the Project

Create an environment and install the dependencies:

```bash
pip install -r requirements.txt
```

Open the main notebook in Jupyter, JupyterLab, or VS Code:

```bash
jupyter lab main.ipynb  # if JupyterLab is installed
```

The notebook is organized as:

1. data loading and preprocessing,
2. simulated SV truth set,
3. KSC Gibbs sampler,
4. NUTS via PyMC,
5. flowMC,
6. sampler comparison metrics,
7. conclusion and limitations.

Sampler outputs are saved as compressed `.npz` files under `samples/`. The comparison step expects saved sampler results to contain chain-shaped draws for parameters and selected latent summaries such as `h_mid` and `h_last`.

Run the tests with:

```bash
python -m pytest -q
```

## Main Takeaways

- KSC Gibbs is the strongest specialized sampler for this baseline SV model. It exploits the state-space structure and jointly samples the latent volatility path with FFBS.
- NUTS is the strongest general-purpose baseline. It is slower than KSC but targets the model directly and gives clear diagnostics.
- flowMC is the most experimental method. The local sampler can be tuned to produce useful draws, but the independent global flow proposal is difficult in the full latent-state posterior.
- The Gaussian SV model captures persistent volatility, but it does not fully capture the tail behavior of real financial returns. VaR exceedance behavior suggests that heavy-tailed errors, leverage effects, jumps, or regime changes would be useful extensions.

## Limitations and Future Work

The simulation study uses a small number of simulated paths, so it is useful for implementation validation but not enough for a full coverage or robustness study. A repeated-simulation experiment would give a more stable picture of bias, coverage, and runtime variability.

The model is also intentionally simple. It captures persistent latent volatility but omits common financial-return features such as leverage effects, jumps, heavy-tailed observation noise, and regime shifts.

For flowMC, the main next step is structural rather than just running longer. Better designs may use blocked latent updates, FFBS-style moves, parameter-only global proposals, or conditional flows that do not try to learn the full high-dimensional joint posterior at once.

## References

Abril-Pla, O., Andreani, V., Carroll, C., Dong, L., Fonnesbeck, C. J., Kochurov, M., Kumar, R., Lao, J., Luhmann, C. C., Martin, O. A., Osthege, M., Vieira, R., Wiecki, T., & Zinkov, R. (2023). PyMC: A modern, and comprehensive probabilistic programming framework in Python. *PeerJ Computer Science, 9*, e1516. https://doi.org/10.7717/peerj-cs.1516

Carter, C. K., & Kohn, R. (1994). On Gibbs sampling for state space models. *Biometrika, 81*(3), 541-553.

Durkan, C., Bekasov, A., Murray, I., & Papamakarios, G. (2019). Neural spline flows. *Advances in Neural Information Processing Systems, 32*.

Hoffman, M. D., & Gelman, A. (2014). The No-U-Turn sampler: Adaptively setting path lengths in Hamiltonian Monte Carlo. *Journal of Machine Learning Research, 15*(47), 1593-1623.

Kim, S., Shephard, N., & Chib, S. (1998). Stochastic volatility: Likelihood inference and comparison with ARCH models. *The Review of Economic Studies, 65*(3), 361-393.

Kumar, R., Carroll, C., Hartikainen, A., & Martin, O. (2019). ArviZ: A unified library for exploratory analysis of Bayesian models in Python. *Journal of Open Source Software, 4*(33), 1143. https://doi.org/10.21105/joss.01143

Kupiec, P. H. (1995). Techniques for verifying the accuracy of risk measurement models. *The Journal of Derivatives, 3*(2), 73-84. https://doi.org/10.3905/jod.1995.407942

Wong, K. W. K., Gabrié, M., & Foreman-Mackey, D. (2023). flowMC: Normalizing flow enhanced sampling package for probabilistic inference in JAX. *Journal of Open Source Software, 8*(83), 5021. https://doi.org/10.21105/joss.05021
