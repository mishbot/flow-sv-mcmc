import numpy as np

# General
SEED      = 42
MODAL_GPU = 'B200'


# Step 1 - Data
STOCK_LIST = ["^GSPC", "AAPL"]
START_DATE = "2016-01-01"
END_DATE   = "2026-01-01"


# Step 2 - Simulation truths (from KSC 1998's empirical fit)
MU_TRUE        = -9.0
PHI_TRUE       = 0.97
SIGMA_ETA_TRUE = 0.15
T_SIM          = 2500   # simulated sample size


# Step 3 - KSC priors and mixture constants
PRIOR_MU_VAR          = 100.0         # mu is N(0, 100)
PRIOR_PHI_BETA        = (20.0, 1.5)   # (phi+1)/2 is Beta(20, 1.5)
PRIOR_SIGMA_ETA_GAMMA = (2.5, 0.025)  # sigma_eta^-2 is Gamma(2.5, 0.025)

KSC_Q  = np.array([0.00730, 0.10556, 0.00002, 0.04395, 0.34001, 0.24566, 0.25750])
KSC_M  = np.array([-10.12999, -3.97281, -8.56686, 2.77786, 0.61942, 1.79518, -1.08819])
KSC_V2 = np.array([5.79596, 2.61369, 5.17950, 0.16735, 0.64009, 0.34023, 1.26261])

KSC_N_CHAINS = 4
KSC_N_ITER   = 30000
KSC_N_BURN   = 5000


# Step 4 - NUTS via PyMC
NUTS_CHAINS        = 4
NUTS_DRAWS         = 2000
NUTS_TUNE          = 2000
NUTS_TARGET_ACCEPT = 0.95


# Step 5 - Flow-Augmented MCMC
FLOW_N_CHAINS     = 64
FLOW_N_HIDDEN     = 64
FLOW_N_LOCAL      = 100     # local MALA steps between flow-training/global rounds
FLOW_N_GLOBAL     = 1       # global accept is near zero in the final runs
FLOW_N_TRAINING   = 60
FLOW_N_PRODUCTION = 100
FLOW_N_EPOCHS     = 3
FLOW_MAX_STORED_DRAWS = 1000

FLOW_LOCAL_KERNEL = "mala"
FLOW_INIT_THETA   = [None, 0.97, 0.15]               # initial values for theta, none for mu (calculated during execution)
FLOW_THETA0_SCALE = [0.03, 0.003, 0.02]              # init_pos scale for (mu, atanh(phi), log(sigma_eta))
FLOW_GRW_STEPSIZE = [0.003, 0.003, 0.003, 0.025]     # optional GRW step size for (mu, atanh(phi), log(sigma_eta), z)
FLOW_MALA_STEPSIZE = [0.002, 0.006, 0.003, 0.012]    # MALA dt for (mu, atanh(phi), log(sigma_eta), z)
FLOW_LOCAL_THINNING = 4


_GROUPS = {
    "General":             ["SEED", "MODAL_GPU"],
    "Step 1 (Data)":       ["STOCK_LIST", "START_DATE", "END_DATE"],
    "Step 2 (Simulation)": ["MU_TRUE", "PHI_TRUE", "SIGMA_ETA_TRUE", "T_SIM"],
    "Step 3 (KSC)":        ["PRIOR_MU_VAR", "PRIOR_PHI_BETA", "PRIOR_SIGMA_ETA_GAMMA",
                            "KSC_Q", "KSC_M", "KSC_V2", "KSC_N_CHAINS",
                            "KSC_N_ITER", "KSC_N_BURN"],
    "Step 4 (NUTS)":       ["NUTS_CHAINS", "NUTS_DRAWS", "NUTS_TUNE", "NUTS_TARGET_ACCEPT"],
    "Step 5 (Flow)":       ["FLOW_N_CHAINS", "FLOW_N_HIDDEN", "FLOW_N_LOCAL",
                            "FLOW_N_GLOBAL", "FLOW_N_TRAINING", "FLOW_N_PRODUCTION",
                            "FLOW_N_EPOCHS", "FLOW_MAX_STORED_DRAWS", "FLOW_LOCAL_KERNEL",
                            "FLOW_INIT_THETA", "FLOW_THETA0_SCALE", "FLOW_GRW_STEPSIZE",
                            "FLOW_MALA_STEPSIZE", "FLOW_LOCAL_THINNING"],
}

def _fmt(val):
    if isinstance(val, np.ndarray):
        return f"array shape={val.shape}, mean={val.mean():+.4f}"
    return repr(val)

def print_constants():
    """Print all project constants, grouped by section."""
    g = globals()
    for idx, (section, names) in enumerate(_GROUPS.items()):
        present = [n for n in names if n in g]
        if not present:
            continue
        print(f"\n{section}" if idx > 0 else section)
        print("─" * 60)
        for name in present:
            print(f"  {name:25s} = {_fmt(g[name])}")
    print()

print_constants()
