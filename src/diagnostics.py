import numpy as np
import pandas as pd
import arviz as az
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf

try:
    from IPython.display import display
except ImportError:  # pragma: no cover
    def display(obj):
        print(obj)


PARAM_VARS = ["mu", "phi", "sigma_eta"]
DIAG_VARS = ["mu", "phi", "sigma_eta", "h_mid", "h_last"]

LATEX_VAR_LABELS = {
    "mu": r"$\mu$",
    "phi": r"$\phi$",
    "sigma_eta": r"$\sigma_\eta$",
    "h_mid": r"$h_{T/2}$",
    "h_last": r"$h_T$",
    "h": r"$h_t$",
    "y": r"$y_t$",
}


def _latex_var_label(variable):
    """Return a plotting label without changing result/table variable names."""
    variable = str(variable)
    if variable.startswith("$") and variable.endswith("$"):
        return variable
    return LATEX_VAR_LABELS.get(variable, fr"${variable}$")


def _latex_square_label(variable):
    label = _latex_var_label(variable)
    return label[:-1] + r"^2$" if label.startswith("$") and label.endswith("$") else f"{label}^2"


def _arviz_labeller():
    """Use LaTeX variable names in ArviZ plots when supported."""
    try:
        return az.labels.MapLabeller(var_name_map=LATEX_VAR_LABELS)
    except AttributeError:
        return None


def sanity_plots(y, title, var):
    """Plot basic time-series diagnostics for returns or latent volatility."""
    y = np.asarray(y)

    fig, axes = plt.subplots(2, 2, figsize=(12, 6))

    axes[0, 0].plot(y, linewidth=0.5)
    axes[0, 0].set_title("Log-Returns Over Time")
    axes[0, 0].set_xlabel(r"$t$")
    axes[0, 0].set_ylabel(_latex_var_label(var))

    axes[0, 1].hist(y, bins=80, density=True, alpha=0.6)
    xx = np.linspace(y.min(), y.max(), 200)
    axes[0, 1].plot(
        xx,
        np.exp(-xx**2 / (2 * y.var())) / np.sqrt(2 * np.pi * y.var()),
        lw=1.5,
    )
    axes[0, 1].set_title("Histogram")

    plot_acf(y, ax=axes[1, 0], title=f"ACF of {_latex_var_label(var)}", lags=50, fft=True)
    plot_acf(y**2, ax=axes[1, 1], title=f"ACF of {_latex_square_label(var)}", lags=50, fft=True)

    fig.suptitle(title)
    plt.tight_layout()
    plt.show()


def result_to_idata(result, var_names=DIAG_VARS):
    """Convert a sampler result dict with chain-shaped arrays to ArviZ."""
    posterior = {
        key: np.asarray(result[key])
        for key in var_names
        if key in result
    }
    if not posterior:
        raise ValueError("No requested posterior variables are present in result.")
    return az.from_dict(posterior=posterior)


def posterior_recovery_table(result, truth):
    """Summarize posterior recovery against simulated truth."""
    rows = []
    for key, true_val in truth.items():
        x = np.asarray(result[key]).reshape(-1)
        lo, hi = np.quantile(x, [0.025, 0.975])
        rows.append({
            "param": key,
            "mean": x.mean(),
            "sd": x.std(),
            "2.5%": lo,
            "97.5%": hi,
            "truth": true_val,
            "in CI": lo <= true_val <= hi,
        })
    return pd.DataFrame(rows).set_index("param")


def ess_per_second(result, idata=None, var_names=DIAG_VARS):
    """Compute bulk ESS per second for variables present in result."""
    elapsed = result.get("elapsed_sec")
    if elapsed is None or elapsed <= 0:
        raise ValueError("result must contain positive elapsed_sec.")

    if idata is None:
        idata = result_to_idata(result, var_names=var_names)

    present_vars = [key for key in var_names if key in result]
    summary = az.summary(idata, var_names=present_vars)
    return summary["ess_bulk"] / elapsed


def display_sampler_diagnostics(
    label,
    result,
    truth=None,
    accept_key=None,
    var_names=DIAG_VARS,
    trace_vars=PARAM_VARS,
    show_ess_per_sec=True,
    show_trace=True,
):
    """Display recovery, ArviZ diagnostics, timing, and trace plots."""
    present_vars = [key for key in var_names if key in result]
    idata = result_to_idata(result, var_names=present_vars)

    print(label)

    if truth is not None:
        display(posterior_recovery_table(result, truth))

    summary = az.summary(idata, var_names=present_vars)
    display(summary)

    if show_ess_per_sec and result.get("elapsed_sec") is not None:
        display((summary["ess_bulk"] / result["elapsed_sec"]).rename("ess_bulk_per_sec"))

    if accept_key is not None and accept_key in result:
        print(f"{accept_key}:", result[accept_key])

    if "divergences" in result:
        print("Divergences:", result["divergences"])

    if result.get("elapsed_sec") is not None:
        print("Elapsed seconds:", result["elapsed_sec"])

    trace_vars = [key for key in trace_vars if key in present_vars]
    if show_trace and trace_vars:
        labeller = _arviz_labeller()
        plot_kwargs = {"labeller": labeller} if labeller is not None else {}
        axes = az.plot_trace(idata, var_names=trace_vars, **plot_kwargs)
        fig = axes.ravel()[0].figure
        fig.suptitle(f"Distribution and Trace for {label}", fontsize=14, y=1.02)
        plt.tight_layout()
        plt.show()

    return idata


def display_flow_diagnostics(label, result, truth=None):
    """Display shared diagnostics plus flowMC acceptance/training diagnostics."""
    idata = display_sampler_diagnostics(label, result, truth=truth)

    print("local accept:", np.nanmean(result["local_accept_prod"]))
    print("global accept:", np.nanmean(result["global_accept_prod"]))
    print("final loss/dim:", result["loss"][-1] / result["n_dim"])

    return idata


def plot_latent_recovery(result, h_true, title="Latent Volatility Recovery"):
    """Plot posterior mean and credible interval for a simulated latent path."""
    h_true = np.asarray(h_true)
    h_mean = np.asarray(result["h_mean"])
    h_ci = np.asarray(result["h_ci"])

    coverage = np.mean((h_true >= h_ci[0]) & (h_true <= h_ci[1]))
    rmse = np.sqrt(np.mean((h_mean - h_true) ** 2))

    print(f"h coverage: {coverage:.3f}")
    print(f"h RMSE:     {rmse:.3f}")

    plt.figure(figsize=(12, 4))
    plt.plot(h_true, color="black", lw=1, label=r"True $h_t$")
    plt.plot(h_mean, color="tab:blue", lw=1, label="Posterior Mean")
    plt.fill_between(
        np.arange(h_true.shape[0]),
        h_ci[0],
        h_ci[1],
        color="tab:blue",
        alpha=0.2,
        label="95% CI",
    )
    plt.title(title)
    plt.xlabel(r"$t$")
    plt.ylabel(r"$h_t$")
    plt.legend()
    plt.tight_layout()
    plt.show()

    return coverage, rmse
