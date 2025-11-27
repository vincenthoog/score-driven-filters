"""
Generic analysis script for the score-driven filter.

Usage
-----
1. Prepare your data as a pandas DataFrame `Y_df` with:
   - rows   = time,
   - columns = cross-sectional units (assets, sectors, regions, etc.).

2. Call `run_analysis(Y_df, method='explicit', ...)` or
   `run_analysis(Y_df, method='implicit', ...)` from your own script.
   
3. Optionally, if you have the Fama-French CSVs, you can use
   `get_data_and_W` to load Fama-French portfolios and then
   pass the resulting DataFrame into `run_analysis` as a test.

This script will:
   - build a simple spatial weight matrix W from correlations of Y_df,
   - build a simple design matrix X (intercept only); regressors can be added,
   - fit either the explicit or implicit score-driven filter via
     their `fit_and_evaluate` functions,
   - print basic diagnostics and make a couple of example plots.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Explicit and implicit frameworks
from explicit_filter import fit_and_evaluate as explicit_fit
from implicit_filter import fit_and_evaluate as implicit_fit

# Optional: if get_data_and_W.py is present, we can reuse its helpers
try:
    from get_data_and_W import get_data, build_W_from_corr
except ImportError:
    get_data = None
    build_W_from_corr = None


# =============================================================================
# Helpers to build X - add regressors here if needed
# =============================================================================
def build_design_matrix(Y: np.ndarray, add_intercept: bool = True) -> np.ndarray:
    """
    Build a design matrix X of shape (T, R, p+1).

    For most use cases, an intercept per series is enough, so by default this
    returns a single-column design with ones.

    Parameters
    ----------
    Y : np.ndarray
        Data array of shape (T, R).
    add_intercept : bool, default True
        If True, include a constant regressor per series.

    Returns
    -------
    X : np.ndarray
        Array of shape (T, R, p+1). With add_intercept=True, p=0, so X has
        shape (T, R, 1).
    """
    T, R = Y.shape
    if add_intercept:
        X = np.ones((T, R, 1), dtype=float)
    else:
        # no regressors; shape (T, R, 0)
        X = np.zeros((T, R, 0), dtype=float)
    return X


# =============================================================================
# High-level analysis function
# =============================================================================
def run_analysis(
    Y_df: pd.DataFrame,
    method: str = "explicit",          # 'explicit' or 'implicit'
    scaling: str = "invFisher",        # scaling method
    start_mode: str = "standard",
    k_neighbors: int = 6,
    min_corr: float = 0.10,
    maxiter: int = 1500,
):
    """
    Run the score-driven filter (explicit or implicit) on user-supplied data.

    Parameters
    ----------
    Y_df : pd.DataFrame
        T x R DataFrame of observations (e.g. log RV, returns, etc.).
    method : {'explicit', 'implicit'}, default 'explicit'
        Which framework to use:
            - 'explicit' : explicit score-driven filter
            - 'implicit' : implicit score-driven filter
    scaling : {'gasp', 'invFisher', 'sqrtInvFisher', 'identity', 'invHessian', 
               'ewma', 'local_hess'}
        Score-scaling choice (passed to the chosen filter).
    start_mode : {'standard', 'local_hess', 'ewma', 'inv_hessian'}
        Filter / scaling family variant (depending on implementation).
    k_neighbors : int, default 6
        Number of nearest neighbours to keep in W per row when building W from
        correlations.
    min_corr : float, default 0.10
        Minimum correlation threshold passed to `build_W_from_corr`.
    maxiter : int, default 1500
        Maximum number of L-BFGS-B iterations for the optimizer.

    Returns
    -------
    results : dict
        Output from the chosen `fit_and_evaluate`, including:
            - results['params_opt']
            - results['mu_in']
            - results['yhat_in']
            - results['avg_log_in']
            - etc.
    """
    # Ensure sorted by time (just in case)
    Y_df = Y_df.sort_index()
    Y = Y_df.to_numpy(dtype=float)
    T, R = Y.shape

    print(f"Data shape: T={T}, R={R}")
    print(f"Method='{method}', scaling='{scaling}', start_mode='{start_mode}'.")

    # ------------------------------------------------------------------
    # Build spatial weight matrix W from correlations
    # ------------------------------------------------------------------
    if build_W_from_corr is None:
        raise ImportError(
            "build_W_from_corr not found. Make sure get_data_and_W.py is "
            "on your PYTHONPATH or in the same directory."
        )

    print("Building spatial weight matrix W from correlations...")
    W_df = build_W_from_corr(
        Y_df,
        k=k_neighbors,
        min_corr=min_corr,
        standardize=True,
        corr_method="pearson",
    )
    W = W_df.to_numpy(dtype=float)

    # ------------------------------------------------------------------
    # Design matrix (intercept-only by default)
    # ------------------------------------------------------------------
    X = build_design_matrix(Y, add_intercept=True)

    # ------------------------------------------------------------------
    # Choose filter implementation
    # ------------------------------------------------------------------
    method = method.lower()
    if method == "explicit":
        fit_fn = explicit_fit
    elif method == "implicit":
        if implicit_fit is None:
            raise ImportError(
                "implicit_filter.fit_and_evaluate could not be imported. "
                "Make sure implicit_filter.py is available and on PYTHONPATH."
            )
        fit_fn = implicit_fit
    else:
        raise ValueError("method must be 'explicit' or 'implicit'")

    # ------------------------------------------------------------------
    # Fit model and evaluate in-sample
    # ------------------------------------------------------------------
    results = fit_fn(
        Y, X, W,
        scaling=scaling,
        start_mode=start_mode,
        save_str=f"start_params_{method}_{scaling}_{start_mode}.pkl",
        maxiter=maxiter,
    )

    # Basic reporting
    opt = results["state"]
    params_opt = results["params_opt"]
    phi, rho1, rho2, theta = params_opt[:4]
    nu = 2 + np.exp(theta)

    print("\n--- Convergence info ---")
    print(f"Converged?   {opt.success}")
    print(f"Final NLL:   {results['final_nll']:.6f}")
    print("\n--- Key parameters (transformed) ---")
    print(f"phi:  {np.tanh(phi):.3f}")
    print(f"rho1: {np.tanh(rho1):.3f}")
    print(f"rho2: {np.tanh(rho2):.3f}")
    print(f"nu:   {nu:.3f}")
    print(f"In-sample avg log-score: {results['avg_log_in']:.8f}")

    # Quick example plots (first few series)
    mu_in = results["mu_in"]
    yhat_in = results["yhat_in"]
    plot_mu(Y, mu_in, plot_K=min(3, R))
    plot_Y(Y, yhat_in, plot_K=min(3, R))

    return results


# =============================================================================
# Simple plotting helpers
# =============================================================================
def plot_mu(Y: np.ndarray, mu_hat: np.ndarray, plot_K: int = 3):
    """Plot latent mu vs observed Y for the first few series."""
    T, R = Y.shape
    for k in range(min(plot_K, R)):
        plt.figure()
        plt.plot(Y[:, k], label="Y")
        plt.plot(mu_hat[:, k], label="mu_hat")
        plt.legend()
        plt.title(f"Latent mu vs Y (series {k})")
        plt.xlabel("time")
        plt.tight_layout()


def plot_Y(Y: np.ndarray, Y_hat: np.ndarray, plot_K: int = 3):
    """Plot observed vs fitted Y for the first few series."""
    T, R = Y.shape
    for k in range(min(plot_K, R)):
        plt.figure()
        plt.plot(Y[:, k], label="Y")
        plt.plot(Y_hat[:, k], label="Y_hat")
        plt.legend()
        plt.title(f"Observed vs fitted Y (series {k})")
        plt.xlabel("time")
        plt.tight_layout()
    plt.show()


# =============================================================================
# Example usage
# =============================================================================
if __name__ == "__main__":
    # -------------------------------------------------------------------------
    # Example 1: Simulated data (works out-of-the-box)
    # -------------------------------------------------------------------------
    np.random.seed(0)
    T, R = 300, 5
    eps = np.random.randn(T, R) * 0.08
    Y_sim = np.zeros((T, R))
    for t in range(1, T):
        Y_sim[t] = 0.8 * Y_sim[t - 1] + eps[t]

    idx = pd.date_range("2000-01-01", periods=T, freq="D")
    cols = [f"series_{j}" for j in range(R)]
    Y_df_sim = pd.DataFrame(Y_sim, index=idx, columns=cols)

    print("\n=== Running analysis on simulated data (explicit) ===")
    _ = run_analysis(
        Y_df_sim,
        method="explicit",
        scaling="invFisher",
        start_mode="standard",
        k_neighbors=6,
        min_corr=0.10,
        maxiter=1000,
    )

    # If implicit_filter is available, you can also test:
    print("\n=== Running analysis on simulated data (implicit) ===")
    _ = run_analysis(
        Y_df_sim,
        method="implicit",
        scaling="invFisher",
        start_mode="standard",
        k_neighbors=6,
        min_corr=0.10,
        maxiter=1000,
    )
