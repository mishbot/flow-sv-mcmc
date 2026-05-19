from pathlib import Path
import arviz as az
import numpy as np
import pandas as pd
from scipy.stats import chi2, wasserstein_distance
import matplotlib.pyplot as plt


COMPARE_VARS = ("mu", "phi", "sigma_eta", "h_mid")

LATEX_VAR_LABELS = {
    "mu": r"$\mu$",
    "phi": r"$\phi$",
    "sigma_eta": r"$\sigma_\eta$",
    "h_mid": r"$h_{T/2}$",
    "h_last": r"$h_T$",
}

SAMPLER_LABELS = {
    "ksc": "KSC",
    "nuts": "NUTS",
    "flow": "flowMC",
}

SAMPLER_COLORS = {
    "KSC": "tab:blue",
    "NUTS": "tab:orange",
    "flowMC": "tab:green",
}


def _latex_var_label(variable):
    """Return a plotting label without changing result/table variable names."""
    return LATEX_VAR_LABELS.get(variable, variable)


def _sampler_label(sampler):
    """Return a display label without changing result/table sampler names."""
    return SAMPLER_LABELS.get(str(sampler), str(sampler).title())


def _sampler_colors(columns):
    return [SAMPLER_COLORS.get(str(column), None) for column in columns]


def _format_bar_axis(ax):
    """Keep grouped-bar labels horizontal and readable."""
    ax.tick_params(axis="x", labelrotation=0)
    for label in ax.get_xticklabels():
        label.set_ha("center")


def _result_to_idata(result, var_names):
    """Convert selected chain arrays to an ArviZ InferenceData object."""
    posterior = {
        key: np.asarray(result[key])
        for key in var_names
        if key in result
    }
    if not posterior:
        raise ValueError("No requested posterior variables are present in result.")
    return az.from_dict(posterior=posterior)


def load_sampler_result(path):
    """Load one saved sampler .npz file."""
    path = Path(path)
    result = {}
    with np.load(path, allow_pickle=False) as raw:
        for key in raw.files:
            value = raw[key]
            result[key] = value.item() if value.shape == () else value
    return result


def load_all_sampler_results(sample_dir="samples", datasets=("sim", "gspc", "aapl"),
                             samplers=("ksc", "nuts", "flow")):
    """Load all available sampler result files for the requested datasets."""
    sample_dir = Path(sample_dir)
    out = {}
    for dataset in datasets:
        out[dataset] = {}
        for sampler in samplers:
            path = sample_dir / f"{sampler}_{dataset}.npz"
            if path.exists():
                out[dataset][sampler] = load_sampler_result(path)
    return out


def ess_per_second_table(results, var_names=COMPARE_VARS):
    """Compute bulk ESS, r-hat, and ESS/sec for each sampler and variable."""
    rows = []
    for dataset, by_sampler in results.items():
        for sampler, result in by_sampler.items():
            present_vars = [key for key in var_names if key in result]
            if not present_vars:
                continue

            idata = _result_to_idata(result, var_names=present_vars)
            summary = az.summary(idata, var_names=present_vars, kind="diagnostics")
            elapsed = float(result.get("elapsed_sec", np.nan))

            for variable, row in summary.iterrows():
                ess_bulk = float(row["ess_bulk"])
                rows.append({
                    "dataset": dataset,
                    "sampler": sampler,
                    "variable": variable,
                    "ess_bulk": ess_bulk,
                    "r_hat": float(row["r_hat"]),
                    "elapsed_sec": elapsed,
                    "ess_per_sec": ess_bulk / elapsed if np.isfinite(elapsed) and elapsed > 0 else np.nan,
                })

    return pd.DataFrame(rows)


def wasserstein_distance_table(results, reference="ksc", var_names=COMPARE_VARS):
    """Compute marginal Wasserstein distances to a reference sampler."""
    rows = []

    for dataset, by_sampler in results.items():
        if reference not in by_sampler:
            continue

        ref = by_sampler[reference]
        for sampler, result in by_sampler.items():
            for variable in var_names:
                if variable not in ref or variable not in result:
                    continue
                distance = 0.0 if sampler == reference else wasserstein_distance(
                    np.asarray(ref[variable]).reshape(-1),
                    np.asarray(result[variable]).reshape(-1),
                )
                rows.append({
                    "dataset": dataset,
                    "sampler": sampler,
                    "variable": variable,
                    f"wasserstein_to_{reference}": distance,
                })

    return pd.DataFrame(rows)


def posterior_predictive_var(result, horizon, tail_prob=0.01, n_paths=50_000, seed=42):
    """Simulate posterior predictive cumulative losses and return VaR."""
    rng = np.random.default_rng(seed)

    mu = np.asarray(result["mu"]).reshape(-1)
    phi = np.asarray(result["phi"]).reshape(-1)
    sigma_eta = np.asarray(result["sigma_eta"]).reshape(-1)
    h_last = np.asarray(result["h_last"]).reshape(-1)

    draw_idx = rng.integers(0, mu.size, size=n_paths)
    mu = mu[draw_idx]
    phi = phi[draw_idx]
    sigma_eta = sigma_eta[draw_idx]
    h = h_last[draw_idx]

    cumulative_return = np.zeros(n_paths, dtype=np.float64)
    for _ in range(horizon):
        h = mu + phi * (h - mu) + sigma_eta * rng.standard_normal(n_paths)
        y = np.exp(h / 2.0) * rng.standard_normal(n_paths)
        cumulative_return += y

    losses = -cumulative_return
    return float(np.quantile(losses, 1.0 - tail_prob))


def var_forecast_table(results, datasets=("gspc",), horizons=(1, 10),
                       tail_probs=(0.01, 0.05), n_paths=50_000, seed=42):
    """Build VaR forecast rows across samplers, horizons, and tail levels."""
    rows = []

    for d_i, dataset in enumerate(datasets):
        if dataset not in results:
            continue
        for s_i, (sampler, result) in enumerate(results[dataset].items()):
            if "h_last" not in result:
                continue
            for horizon in horizons:
                for tail_prob in tail_probs:
                    var = posterior_predictive_var(
                        result,
                        horizon=horizon,
                        tail_prob=tail_prob,
                        n_paths=n_paths,
                        seed=seed + 1000 * d_i + 100 * s_i + 10 * horizon + int(tail_prob * 1000),
                    )
                    rows.append({
                        "dataset": dataset,
                        "sampler": sampler,
                        "horizon": horizon,
                        "tail_prob": tail_prob,
                        "var": var,
                        "n_paths": n_paths,
                    })

    return pd.DataFrame(rows)


def rolling_losses(returns, horizon):
    """Convert returns to one-day or rolling multi-day losses."""
    returns = np.asarray(returns)
    if horizon == 1:
        return -returns
    return -np.convolve(returns, np.ones(horizon), mode="valid")


def kupiec_test(exceedances, tail_prob):
    """Run Kupiec's unconditional coverage test for VaR exceedances."""
    exceedances = np.asarray(exceedances, dtype=bool)
    n = int(exceedances.size)
    x = int(exceedances.sum())

    def _log_lik(p):
        if p <= 0:
            return 0.0 if x == 0 else -np.inf
        if p >= 1:
            return 0.0 if x == n else -np.inf
        return (n - x) * np.log1p(-p) + x * np.log(p)

    p_hat = x / n if n else np.nan
    lr_uc = -2.0 * (_log_lik(tail_prob) - _log_lik(p_hat)) if n else np.nan
    p_value = float(chi2.sf(lr_uc, df=1)) if np.isfinite(lr_uc) else np.nan

    return {
        "n_obs": n,
        "n_exceed": x,
        "expected_exceed": n * tail_prob,
        "exceed_rate": p_hat,
        "lr_uc": lr_uc,
        "p_value": p_value,
    }


def kupiec_backtest_table(var_df, returns_by_dataset):
    """Backtest each VaR forecast against realized losses."""
    rows = []

    for row in var_df.to_dict("records"):
        dataset = row["dataset"]
        if dataset not in returns_by_dataset:
            continue

        losses = rolling_losses(returns_by_dataset[dataset], int(row["horizon"]))
        test = kupiec_test(losses > row["var"], row["tail_prob"])
        rows.append({**row, **test})

    return pd.DataFrame(rows)


def plot_ess_per_second(ess_df, dataset):
    """Plot ESS/sec by sampler for one dataset."""
    data = ess_df[ess_df["dataset"] == dataset]
    pivot = data.pivot(index="variable", columns="sampler", values="ess_per_sec")
    pivot = pivot.rename(index=_latex_var_label)
    pivot = pivot.rename(columns=_sampler_label)
    ax = pivot.plot(kind="bar", figsize=(9, 4), color=_sampler_colors(pivot.columns))
    ax.set_title(f"ESS/sec by Sampler: {dataset.upper()}")
    ax.set_ylabel("Bulk ESS/sec")
    ax.set_xlabel("")
    ax.legend(title="Sampler")
    _format_bar_axis(ax)
    plt.tight_layout()
    plt.show()


def plot_wasserstein(wasserstein_df, dataset, reference="ksc"):
    """Plot Wasserstein distances to the reference sampler for one dataset."""
    value_col = f"wasserstein_to_{reference}"
    data = wasserstein_df[(wasserstein_df["dataset"] == dataset)
                          & (wasserstein_df["sampler"] != reference)]
    pivot = data.pivot(index="variable", columns="sampler", values=value_col)
    pivot = pivot.rename(index=_latex_var_label)
    pivot = pivot.rename(columns=_sampler_label)
    ax = pivot.plot(kind="bar", figsize=(9, 4), color=_sampler_colors(pivot.columns))
    ax.set_title(f"Wasserstein Distance to {reference.upper()}: {dataset.upper()}")
    ax.set_ylabel("Distance")
    ax.set_xlabel("")
    ax.legend(title="Sampler")
    _format_bar_axis(ax)
    plt.tight_layout()
    plt.show()


def plot_kupiec_exceedance_rates(kupiec_df, dataset):
    """Plot realized VaR exceedance rates against expected tail levels."""
    data = kupiec_df[kupiec_df["dataset"] == dataset].copy()
    if data.empty:
        return

    data = data.sort_values(["horizon", "tail_prob", "sampler"])
    data["sampler_label"] = data["sampler"].map(_sampler_label)
    data["setting"] = data.apply(
        lambda row: f"{int(row['horizon'])}-Day\n{row['tail_prob']:.0%} Tail",
        axis=1,
    )

    setting_order = data.drop_duplicates(["horizon", "tail_prob"])["setting"]
    pivot = data.pivot_table(
        index="setting",
        columns="sampler_label",
        values="exceed_rate",
        aggfunc="mean",
    ).reindex(setting_order)

    ax = pivot.plot(kind="bar", figsize=(10, 4), color=_sampler_colors(pivot.columns))

    expected = (
        data.drop_duplicates("setting")
        .set_index("setting")
        .loc[pivot.index, "tail_prob"]
        .to_numpy()
    )
    ax.scatter(
        np.arange(len(pivot)),
        expected,
        color="black",
        marker="_",
        s=120,
        label="Expected",
    )
    ax.set_title(f"Kupiec Exceedance Rates: {dataset.upper()}")
    ax.set_ylabel("Exceedance Rate")
    ax.set_xlabel("")
    ax.legend(title="Sampler")
    _format_bar_axis(ax)
    plt.tight_layout()
    plt.show()
