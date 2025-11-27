"""
analyse_forecast.py

Example script: in-sample and out-of-sample forecast evaluation
using Fama–French-style 49 industry excess returns.

What this script does
---------------------
1. Loads daily Fama–French excess returns via `get_data_and_W`.
2. Aggregates to *monthly realized variance* by summing daily squared returns.
3. Takes the log of realized variance to obtain `y_vol` (T x 49).
4. Splits the sample into:
     - in-sample (first `cT_in` months),
     - out-of-sample (remaining months).
5. Builds:
     - X: intercept-only design matrix,
     - W: cross-sectional weight matrix from in-sample correlations.
6. Runs the unified explicit score-driven framework (via `explicit_analyse_oos`)
   on the in-sample segment and evaluates forecasts out-of-sample.
7. Computes:
     - in-sample and out-of-sample MSE,
     - baseline random-walk and mean benchmarks,
     - per-series OOS gains vs RW,
     - OOS average log-score,
     - Diebold–Mariano (DM) t-statistic and cross-sectional t-statistics.
8. Produces a couple of simple plots for Y vs Y_hat and the latent mu.

This is meant as an **example / demo script**, not a general-purpose API.
To use the framework on your own data, copy the structure (how Y, X, and W
are built and how `explicit_analyse_oos` is called) and replace the
Fama–French data-loading part.

Requirements
------------
- Fama–French CSV files (49 Industry Portfolios, etc.) in the working
  directory, as expected by `get_data_and_W.get_data`.
- The `explicit_filter.py` implementation with `explicit_analyse_oos`.
"""

import pickle
import numpy as np
import pandas as pd

from get_data_and_W import get_data, build_W_from_corr
from explicit_filter import explicit_analyse_oos


# ============================================================
# 1. Import data (Fama–French style example)
# ============================================================
# get_data(False, "daily") is a helper that reads Fama–French CSV files
# and returns daily value-weighted excess returns for 49 industries.
vw_xs_daily, _ = get_data(False, "daily")
assert isinstance(vw_xs_daily.index, pd.DatetimeIndex)

# Monthly realized variance: sum of daily squared excess returns within month
rv_m = (vw_xs_daily**2).resample("M").sum()

# Take logs (add small epsilon for numerical stability)
y_vol = np.log(rv_m + 1e-12)

print(rv_m.index.min(), "->", rv_m.index.max(), rv_m.shape)
print("Quantiles of log-RV:", np.nanquantile(y_vol.values, [0.01, 0.5, 0.99]))


# ============================================================
# 2. Global config and sample split
# ============================================================

# Dimensions: cT = time, cR = cross-sectional units (49 industries)
cT, cR = y_vol.shape
p = 0  # number of non-intercept regressors

# Split into in-sample and out-of-sample horizons
cT_in = 400
cT_out = cT - cT_in

# Design matrix X: intercept-only (shape T x R x (p+1) = T x R x 1)
X_total = np.ones((cT, cR, p + 1), dtype=np.float64)
X_in = X_total[:cT_in, :, :]
X_out = X_total[cT_in:, :, :]

# Response Y: log realized variance
Y_total = y_vol.to_numpy()
Y_in = Y_total[:cT_in, :]
Y_out = Y_total[cT_in:, :]


# ============================================================
# 3. Cross-sectional weight matrix W
# ============================================================

# Build W from in-sample correlations:
# - nearest-neighbour graph with k=8 neighbours per row,
# - min_corr=0.0 -> keep all positive correlations,
# - W is row-stochastic.
W_df = build_W_from_corr(y_vol.iloc[:cT_in], k=8, min_corr=0.0)
W = W_df.to_numpy()


# ============================================================
# 4. Model choices for the unified explicit framework
# ============================================================
method = "implicit"  # 'explicit' or 'implicit'

# Score scaling:
#   'gasp'       : original GASP scaling
#   'invFisher'  : inverse Fisher scaling
#   'sqrtInvFisher' : square-root inverse Fisher scaling
#   'identity'   : gradient / identity scaling
#   'invHessian' : inverse-Hessian-type scaling
#   'ewma'       : EWMA-of-Hessian scaling
#   'local_hess' : local-Hessian scaling
scaling = "gasp"

# Filter variant (start_mode):
#   'standard'    -> standard explicit filter
#   'local_hess'  -> local-Hessian filter (kappa_h, delta, gamma)
#   'ewma'        -> EWMA-of-Hessian filter (delta, gamma)
#   'inv_hessian' -> inverse-Hessian-type filter (delta)
start_mode = "inv_hessian"


# ============================================================
# 5. Analyse (in-sample + out-of-sample)
# ============================================================

print(f"Analyzing the {method} filter with scaling={scaling}, start_mode={start_mode}")
print("")

# explicit_analyse_oos:
#   - fits the explicit score-driven model on (Y_in, X_in, W),
#   - computes in-sample metrics,
#   - rolls forward out-of-sample on (Y_out, X_out) with fixed parameters,
#   - returns a dict with parameters, state, fitted values, etc.
model_res = explicit_analyse_oos(
    Y_in,
    X_in,
    mu_in=None,
    Y_out=Y_out,
    X_out=X_out,
    mu_out=None,
    W=W,
    scaling=scaling,
    start_mode=start_mode,
    cb_start=False,
    cb_general=False,
    save_str="store_results.pkl",
    maxiter=1500,
)

opt_params = model_res["params"]
phi, rho1, rho2, nu, beta = opt_params[:5]
mse_y_in = model_res["metrics"]["mse_y_in"]
mse_y_out = model_res["metrics"]["mse_y_out"]

# Save the full result dict for further analysis / plotting
final_save = f"forecast_results_{method}_{scaling}.pkl"
with open(final_save, "wb") as f:
    pickle.dump(model_res, f)

opt = model_res["state"]
print("\n\nSaved optimizer results to", repr(final_save), "\n")
print("-----------------------Convergence------------------------")
print(f"Converged?: {opt.success}\n")
print("----------------------Optimal Point-----------------------")
print(f"Optimal function value: {opt.fun_val}")
print(f"phi:  {np.tanh(phi):.3f}, \t nu: {2 + np.exp(nu):.3f}")
print(f"rho1: {np.tanh(rho1):.3f}, \t rho2: {np.tanh(rho2):.3f}")
print(f"beta: {round(beta, 3)}")
print(f"MSE of Y, in-sample:  {mse_y_in:.5f}")
print(f"MSE of Y, out-of-sample: {mse_y_out:.5f}")


# ============================================================
# 6. Quick plots: OOS Y vs Y_hat, and mu_hat
# ============================================================

import matplotlib.pyplot as plt


def plot_Y(Y, Y_hat, plot_K=3):
    """
    Simple plot of observed vs predicted Y for a few series.
    """
    for K in range(min(plot_K, Y.shape[1])):
        plt.figure(f"Y_{K}")
        plt.plot(Y[:, K], label="Y")
        plt.plot(Y_hat[:, K], label="Y_hat")
        plt.legend()
        plt.title(f"Observed vs fitted Y (OOS, series {K})")
        plt.show()


def plot_mu(mu_hat, plot_K=3):
    """
    Plot latent mu_hat for a few series (OOS path).
    """
    for K in range(min(plot_K, mu_hat.shape[1])):
        plt.figure(f"mu_{K}")
        plt.plot(mu_hat[:, K], label="mu_hat")
        plt.legend()
        plt.title(f"OOS latent mu (series {K})")
        plt.show()


plot_Y(Y_out, model_res["y_hat_out"], 2)
plot_mu(model_res["mu_out_hat"], 2)


# ============================================================
# 7. Baselines: RW, window mean, and Diebold–Mariano test
# ============================================================

def mse(a, b):
    """
    Mean squared error between two arrays (ignoring NaN patterns).
    """
    return float(np.nanmean((a - b) ** 2))


split = cT_in
Y_te = Y_out
yhat_te = model_res["y_hat_out"]
mse_model = mse(Y_te, yhat_te)

# Random-walk baseline: predict Y_t by last observed Y_{t-1}
yhat_rw = Y_total[split - 1 : -1]
mse_rw = mse(Y_total[split:], yhat_rw)

# Window-mean baseline: in-sample cross-sectional mean, used as constant forecast
mu_tr = np.nanmean(Y_total[:split], axis=0, keepdims=True)
mse_mean = mse(Y_total[split:], mu_tr)

print("MSE(model):", mse_model)
print("MSE(RW):   ", mse_rw, "  gain vs RW (%):  ", 100 * (1 - mse_model / mse_rw))
print(
    "MSE(mean): ",
    mse_mean,
    "  gain vs mean (%):",
    100 * (1 - mse_model / mse_mean),
)

# --- per-industry OOS gains vs RW ---
# MSE per series for model vs random walk
mse_i = np.mean((Y_te - yhat_te) ** 2, axis=0)
yhat_rw = Y_total[split - 1 : -1]
mse_rw_i = np.mean((Y_te - yhat_rw) ** 2, axis=0)
gain_i = 100 * (1 - mse_i / mse_rw_i)
print("Median gain vs RW (%):", float(np.median(gain_i)))
print("IQR gain vs RW (%):", float(np.quantile(gain_i, 0.75) - np.quantile(gain_i, 0.25)))

# --- OOS average log predictive density ---
avg_log_te = float(np.mean(np.array(model_res["llik_out"])))
print("OOS avg log-score (model):", avg_log_te)

e_mod = Y_te - yhat_te
e_rw = Y_te - yhat_rw

# Panel-avg loss differential per month (positive means model better)
dt = np.mean(e_rw**2 - e_mod**2, axis=1)
T_te = dt.shape[0]

# Newey–West SE with lag L (monthly; try L = 3)
L = 3
dbar = np.mean(dt)
gamma0 = np.mean((dt - dbar) ** 2)
S = gamma0
for l in range(1, L + 1):
    w = 1 - l / (L + 1)
    cov = np.mean((dt[l:] - dbar) * (dt[:-l] - dbar))
    S += 2 * w * cov
se = np.sqrt(S / T_te)
t_dm = dbar / se
print("DM t-stat (squared errors):", float(t_dm))

d_i = np.mean((e_rw**2 - e_mod**2), axis=0)  # length R
print("Mean(d_i)=", float(np.mean(d_i)), "Median(d_i)=", float(np.median(d_i)))
# Simple t-stat across industries (not time-robust but informative)
t_cs = np.mean(d_i) / (np.std(d_i, ddof=1) / np.sqrt(d_i.size))
print("Cross-sectional t (49 series):", float(t_cs))

