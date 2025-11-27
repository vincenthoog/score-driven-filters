# determine_start_parameters.py
#
# Unified start-parameter determination for explicit score-driven model:
# - shared JAX-based start LL objective
# - shared coarse grid over rho1 / rho2 + IRLS for beta, lambda
# - flexible start_mode for the final par0 layout:
#     * 'standard'    -> standard scalings (gasp, invFisher, sqrtInvFisher, identity)
#     * 'local_hess'  -> adds local-Hessian parameters [kappa_h, delta_raw, gamma]
#     * 'ewma'        -> adds [delta_raw, gamma]
#     * 'inv_hessian' -> adds [delta]

import time
import pickle
import numpy as np
from scipy.optimize import minimize

import jax
import jax.numpy as jnp
import jax.scipy as jsp
from enum import Enum
from dataclasses import dataclass


# =============================================================================
# Context and enums
# =============================================================================

@dataclass
class StartDetContext:
    """Holds the common data needed for start-parameter determination."""
    Y: np.ndarray
    X: np.ndarray
    W1: np.ndarray
    W2: np.ndarray

    @property
    def T(self) -> int:
        return self.Y.shape[0]

    @property
    def R(self) -> int:
        return self.Y.shape[1]

    @property
    def p(self) -> int:
        # number of regressors minus intercept (so beta has length p+1)
        return self.X.shape[2] - 1


class ScalingCode(Enum):
    """Enumeration of score-scaling choices for the phi/kappa grid search."""
    GASP = 0
    INV_FISHER = 1
    SQRT_INV_FISHER = 2
    IDENTITY = 3


# =============================================================================
# Bounds
# =============================================================================

def construct_bounds(p: int, R: int, kind: str, start_mode: str = "standard"):
    """Construct parameter bounds for different optimization stages.

    Parameters
    ----------
    p : int
        Number of regressors minus intercept (beta has length p+1).
    R : int
        Number of cross-sectional units.
    kind : {'start', 'general'}
        'start'   : bounds for stage-3 start optimization
                    (parameters [eta_r1, eta_r2, theta, beta(a), lambda(b)]).
        'general' : bounds for the full par0 vector used in model estimation.
    start_mode : {'standard', 'local_hess', 'ewma', 'inv_hessian'}
        Controls which extra parameters are present in par0:
        - 'standard'    : no local-Hessian/EWMA extras
        - 'local_hess'  : adds [kappa_h, delta_raw, gamma]
        - 'ewma'        : adds [delta_raw, gamma]
        - 'inv_hessian' : adds [delta]
    """
    if kind == 'start':
        # parameters in stage-3: [eta_r1, eta_r2, theta, beta(a), lam(b)]
        a = 1 + p
        b = R
        lb = np.r_[-4, -4, -4.0, [-5.0] * a, [-5.0] * b]
        ub = np.r_[ 4,  4,  6.0, [ 5.0] * a, [ 5.0] * b]

    elif kind == 'general':
        a = 1 + p
        b = R

        if start_mode == "standard":
            # [phi, rho1, rho2, theta] + beta(a) + lambda(b) + kappa_vec(b)
            lb = np.r_[[ -np.inf]*4,
                       [ -np.inf]*a,
                       [ -np.inf]*b,
                       [ 1e-7   ]*b]
            ub = np.r_[[  np.inf]*4,
                       [  np.inf]*a,
                       [  np.inf]*b,
                       [  np.inf]*b]

        elif start_mode == "local_hess":
            # same as standard + [kappa_h, delta_raw, gamma]
            lb = np.r_[[ -np.inf]*4,
                       [ -np.inf]*a,
                       [ -np.inf]*b,
                       [ 1e-4   ]*b,
                       [ -np.inf, -np.inf, 1e-5]]
            ub = np.r_[[  np.inf]*4,
                       [  np.inf]*a,
                       [  np.inf]*b,
                       [  np.inf]*b,
                       [  np.inf,  np.inf, 0.999999]]

        elif start_mode == "ewma":
            # + [delta_raw, gamma]
            lb = np.r_[[ -np.inf]*4,
                       [ -np.inf]*a,
                       [ -np.inf]*b,
                       [ 1e-4   ]*b,
                       [ -np.inf, 1e-6]]
            ub = np.r_[[  np.inf]*4,
                       [  np.inf]*a,
                       [  np.inf]*b,
                       [  np.inf]*b,
                       [  np.inf, 0.999999]]

        elif start_mode == "inv_hessian":
            # + [delta]
            lb = np.r_[[ -np.inf]*4,
                       [ -np.inf]*a,
                       [ -np.inf]*b,
                       [ 1e-4   ]*b,
                       [ -np.inf]]
            ub = np.r_[[  np.inf]*4,
                       [  np.inf]*a,
                       [  np.inf]*b,
                       [  np.inf]*b,
                       [  np.inf]]

        else:
            raise ValueError(f"Unknown start_mode '{start_mode}'")

    else:
        raise ValueError(f"kind must be 'start' or 'general', got '{kind}'")

    return list(zip(lb, ub))


# =============================================================================
# JAX start-parameter loglikelihood
# =============================================================================

def _opt_start_params_jax(parameters, Y, X, W1, W2, stage, opt1_rho2, opt1_pack):
    """JAX scalar objective for start-parameter search (no mu / kappa yet)."""
    Y = jnp.asarray(Y); X = jnp.asarray(X); W1 = jnp.asarray(W1); W2 = jnp.asarray(W2)
    cT, cR = Y.shape
    p = X.shape[2] - 1

    if stage == 1:
        rho1  = jnp.tanh(parameters[0])
        rho2  = jnp.tanh(opt1_rho2)
        nu    = 2.0 + jnp.exp(parameters[1])
        beta  = parameters[2 : 3 + p].reshape(p + 1, 1)
        lam   = parameters[3 + p : 3 + p + cR].reshape(cR, 1)
    elif stage == 2:
        rho1  = jnp.tanh(opt1_pack[0])
        rho2  = jnp.tanh(parameters[0])
        nu    = 2.0 + jnp.exp(opt1_pack[1])
        beta  = opt1_pack[2 : 3 + p].reshape(p + 1, 1)
        lam   = opt1_pack[3 + p : 3 + p + cR].reshape(cR, 1)
    elif stage == 3:
        rho1  = jnp.tanh(parameters[0])
        rho2  = jnp.tanh(parameters[1])
        nu    = 2.0 + jnp.exp(parameters[2])
        beta  = parameters[3 : 4 + p].reshape(p + 1, 1)
        lam   = parameters[4 + p : 4 + p + cR].reshape(cR, 1)
    else:
        raise ValueError("stage must be 1, 2, or 3")

    lam_inv = jnp.exp(-2.0 * lam).reshape(cR, 1)

    I = jnp.eye(cR)
    Z1 = I - rho1 * W1
    Z2 = I - rho2 * W2

    s1, ld1 = jnp.linalg.slogdet(Z1)
    s2, ld2 = jnp.linalg.slogdet(Z2)
    bad = (s1 <= 0) | (s2 <= 0)
    pen = jnp.where(bad, 1e12, 0.0)

    const = (
        jsp.special.gammaln((nu + cR) / 2.0)
        - jsp.special.gammaln(nu / 2.0)
        - 0.5 * cR * jnp.log(nu)
        - 0.5 * cR * jnp.log(jnp.pi)
        + ld1 + ld2
        - lam.sum()
    )

    Z1Y   = Y @ Z1.T
    Xbeta = (X @ beta).squeeze(-1)
    dev   = Z1Y - Xbeta
    x     = dev @ Z2.T

    q = jnp.sum((x**2) * lam_inv.squeeze(-1)[None, :], axis=1)

    total_ll = cT * const - 0.5 * (nu + cR) * jnp.sum(jnp.log1p(q / nu))
    return -total_ll + pen


_opt_start_params_jax_jit = jax.jit(
    _opt_start_params_jax,
    static_argnames=("stage",)
)


def opt_start_params(parameters, ctx: StartDetContext, stage, opt1, state):
    """NumPy/SciPy-friendly wrapper that calls the JITed JAX scalar objective."""
    rho2_state = state.get('rho2', 0.0)
    opt1_pack = jnp.asarray(opt1) if opt1 is not None else None
    val = _opt_start_params_jax_jit(
        jnp.asarray(parameters),
        ctx.Y, ctx.X, ctx.W1, ctx.W2,
        stage, rho2_state, opt1_pack
    )
    return float(val)


# =============================================================================
# Stage-1 IRLS and coarse grid over rho1 / rho2
# =============================================================================

def _irls_beta_lambda_stage1(ctx: StartDetContext, rho1, iters=5, nu=10.0, eps=1e-12):
    """IRLS for (beta, lambda) at a fixed rho1, used to get a good initial pack."""
    Y, X, W1 = ctx.Y, ctx.X, ctx.W1
    T, R = Y.shape
    p = ctx.p

    Z1Y  = Y - np.tanh(rho1) * (Y @ W1.T)
    beta = np.zeros((p + 1, 1))
    lam  = 0.5 * np.log(np.clip(np.var(Z1Y, axis=0, ddof=1), eps, None)).reshape(R, 1)
    lam_inv = np.exp(-2.0 * lam).reshape(R)

    for _ in range(iters):
        dev = Z1Y - (X @ beta).squeeze(-1)
        q   = np.sum((dev**2) * lam_inv[None, :], axis=1)
        w   = (nu + R) / (nu + q)
        sqrt_w = np.sqrt(w)[:, None]
        s = np.exp(-lam).reshape(R)

        Yw = (Z1Y * s[None, :]) * sqrt_w
        Xw = (X   * s[None, :, None]) * sqrt_w[:, None]

        beta = np.linalg.lstsq(Xw.reshape(T * R, p + 1),
                               Yw.reshape(T * R, 1),
                               rcond=None)[0]
        dev  = Z1Y - (X @ beta).squeeze(-1)
        v    = (w[:, None] * (dev**2)).mean(axis=0)
        lam  = 0.5 * np.log(np.clip(v, eps, None)).reshape(R, 1)
        lam_inv = np.exp(-2.0 * lam).reshape(R)

    return beta.reshape(p + 1), lam.reshape(R), np.log(nu - 2.0)


def _stage1_from_rho1(ctx: StartDetContext, rho1):
    beta, lam, theta = _irls_beta_lambda_stage1(ctx, rho1)
    return np.concatenate([[rho1, theta], beta, lam])


def _coarse_grid_rho1(ctx: StartDetContext, grid=50):
    rhos = np.linspace(-4, 4, grid)

    packs = []
    vals  = []
    state = {'rho2': 0.0}
    for rho1 in rhos:
        pars1 = _stage1_from_rho1(ctx, rho1)
        packs.append(pars1)
        v = opt_start_params(pars1, ctx, 1, None, state)
        vals.append(v)

    i = int(np.argmin(vals))
    return packs[i]


def _coarse_grid_rho2(ctx: StartDetContext, pars1, grid=50):
    rhos = np.linspace(-4, 4, grid)
    state = {'rho2': 0.0}
    vals = []
    for r in rhos:
        v = opt_start_params(np.array([r]), ctx, 2, pars1, state)
        vals.append(v)
    return float(rhos[int(np.argmin(vals))])


def get_starting_params(ctx: StartDetContext, callb: bool, save_str: str):
    """Stage-1+2+3 start parameter optimization (no phi / kappa yet)."""
    cT, cR = ctx.Y.shape
    p = ctx.p

    if callb:
        print('Start parameter optimization stage 1 (coarse+IRLS)...')
    pars1 = _coarse_grid_rho1(ctx, grid=50)

    if callb:
        print('Updating rho2 (coarse scan)...')
    rho2 = _coarse_grid_rho2(ctx, pars1, grid=50)
    if callb:
        print('Updated rho2 to:', np.round(rho2, 4))

    # Stage-3: local L-BFGS-B on full data
    par0_stage3 = np.concatenate([pars1[:1], [rho2], pars1[1:]])
    stage_3_bnds = construct_bounds(p, cR, 'start')

    if callb:
        print('Start parameter optimization stage 3 (local L-BFGS-B)...')
    res3 = minimize(
        opt_start_params, par0_stage3,
        args=(ctx, 3, None, {'rho2': rho2}),
        bounds=stage_3_bnds, method='L-BFGS-B',
        options={'maxiter': 300, 'ftol': 1e-6, 'maxfun': 50000}
    )

    with open(save_str, "wb") as f:
        pickle.dump(res3, f)
    print(f"Saved starting parameters to '{save_str}'")
    return res3


# =============================================================================
# Grid over phi and kappa (vector)
# =============================================================================

def grid_phi_kappa_scalar_vec(ctx: StartDetContext,
                              pars,
                              scaling_code: ScalingCode,
                              phi_grid=21,
                              k_grid=25, k_lo=1e-3, k_hi=30.0):
    """Grid search for phi (scalar) and kappa (scalar -> then vector)."""
    Y, X, W1, W2 = ctx.Y, ctx.X, ctx.W1, ctx.W2
    T, R = Y.shape
    p = ctx.p

    eta_r1, eta_r2, theta = pars[0], pars[1], pars[2]
    beta  = pars[3:4 + p].reshape(p + 1, 1)
    lam   = pars[4 + p:4 + p + R].reshape(R, 1)

    rho1, rho2 = np.tanh(eta_r1), np.tanh(eta_r2)
    nu  = 2.0 + np.exp(theta)
    Z1 = np.eye(R) - rho1 * W1
    Z2 = np.eye(R) - rho2 * W2

    # optional subsample (currently not used)
    Yg, Xg = Y, X

    # precompute dev base and static pieces
    Z1Y   = Yg @ Z1.T
    Xbeta = (Xg @ beta).squeeze(-1)
    base_dev = Z1Y - Xbeta
    lam_inv = np.exp(-2.0 * lam.ravel())
    sqrt_lam_inv = np.sqrt(lam_inv)
    LZ2T = (Z2.T * lam_inv[:, None])

    X_all = Z2 @ base_dev.T
    q_all = (X_all**2 * lam_inv[:, None]).sum(axis=0)
    alph_all = 1.0 + q_all / nu

    if scaling_code is ScalingCode.GASP:
        S0 = (X_all / alph_all[None, :]).T
    elif scaling_code is ScalingCode.INV_FISHER:
        c = (nu + R + 2.0) / (nu * alph_all)
        S0 = (base_dev * c[:, None])
    elif scaling_code is ScalingCode.SQRT_INV_FISHER:
        c0 = np.sqrt((nu + R + 2.0) / (nu + R)) * (nu + R) / nu
        S0 = (c0 * (sqrt_lam_inv[:, None] * X_all) / alph_all[None, :]).T
    elif scaling_code is ScalingCode.IDENTITY:
        c = (nu + R) / (nu * alph_all)
        Av_all = (LZ2T @ X_all).T
        S0 = Av_all * c[:, None]
    else:
        raise ValueError(f"Unsupported ScalingCode '{scaling_code}'")

    s_std = np.sqrt(S0.var(axis=0) + 1e-12)
    s_std[s_std == 0] = 1.0

    phis = np.linspace(0.0, 0.98, phi_grid)
    kappas = np.exp(np.linspace(np.log(k_lo), np.log(k_hi), k_grid))
    PHI, KAP = np.meshgrid(phis, kappas, indexing='xy')
    PHI = PHI.ravel()
    KAP = KAP.ravel()
    G = PHI.size

    mu = np.zeros((R, G))
    ll = np.zeros(G)

    for t in range(base_dev.shape[0]):
        dev_t_col = base_dev[t][:, None]
        dev_all = dev_t_col - mu
        x_all = Z2 @ dev_all
        q = (x_all**2 * lam_inv[:, None]).sum(axis=0)
        alph = 1.0 + q / nu

        if scaling_code is ScalingCode.GASP:
            S = x_all / alph[None, :]
        elif scaling_code is ScalingCode.INV_FISHER:
            c = (nu + R + 2.0) / (nu * alph)
            S = dev_all * c[None, :]
        elif scaling_code is ScalingCode.SQRT_INV_FISHER:
            c0 = np.sqrt((nu + R + 2.0) / (nu + R)) * (nu + R) / nu
            S  = (c0 * (sqrt_lam_inv[:, None] * x_all) / alph[None, :])
        elif scaling_code is ScalingCode.IDENTITY:
            c = (nu + R) / (nu * alph)
            Av = LZ2T @ x_all
            S = Av * c[None, :]
        else:
            raise ValueError(f"Unsupported ScalingCode '{scaling_code}'")

        S_norm = S / s_std[:, None]
        mu = PHI[None, :] * mu + KAP[None, :] * S_norm

        ll += -0.5 * (nu + R) * np.log(alph)

    g_best = int(np.argmax(ll))
    phi_star = float(PHI[g_best])
    kappa_scalar = float(KAP[g_best])

    kap0_vec = kappa_scalar / s_std
    return phi_star, kap0_vec, kappa_scalar


# =============================================================================
# High-level: build par0
# =============================================================================

_SCALING_CODE_MAP = {
    'gasp':           ScalingCode.GASP,
    'invFisher':      ScalingCode.INV_FISHER,
    'sqrtInvFisher':  ScalingCode.SQRT_INV_FISHER,
    'identity':       ScalingCode.IDENTITY,
    'invHessian':     ScalingCode.IDENTITY,  # same code as identity here
}


def build_start_params(Y, X, W1, W2,
                       cb: bool,
                       scaling,
                       start,
                       save_str: str,
                       start_mode: str = "standard"):
    """Unified entry point to build par0 and bounds.

    Parameters
    ----------
    Y, X, W1, W2 : arrays
        Data and spatial weight matrices.
    cb : bool
        If True, print progress information.
    scaling : {str, int, ScalingCode}
        Score-scaling choice: 'gasp', 'invFisher', 'sqrtInvFisher',
        'identity', 'invHessian', an integer code, or a ScalingCode.
    start : OptimizeResult or None
        If not None, reuse the result of get_starting_params; otherwise run it.
    save_str : str
        Path to pickle file for the get_starting_params() result.
    start_mode : {'standard', 'local_hess', 'ewma', 'inv_hessian'}
        Controls which extra parameters appear in par0 (and matching bounds).

    Returns
    -------
    par0 : np.ndarray
        Starting parameter vector.
    bounds : list of (low, high)
        Bounds for the 'general' parameter vector.
    """
    ctx = StartDetContext(Y=Y, X=X, W1=W1, W2=W2)

    t_start = time.time()

    # interpret scaling argument
    if isinstance(scaling, str):
        try:
            scaling_code = _SCALING_CODE_MAP[scaling]
        except KeyError:
            raise ValueError(
                f"Unknown scaling '{scaling}'. "
                f"Available: {list(_SCALING_CODE_MAP.keys())}"
            )
    elif isinstance(scaling, ScalingCode):
        scaling_code = scaling
    else:
        # assume integer code
        scaling_code = ScalingCode(int(scaling))

    print('--- Start parameter optimization ---')
    cT, cR = ctx.Y.shape
    p = ctx.p

    # Stage 1–3 for [rho1, rho2, theta, beta, lambda]
    if start is not None:
        res_start = start
        pars = res_start.x
    else:
        res_start = get_starting_params(ctx, cb, save_str)
        pars = res_start.x

    print('Optimized starting values (stage 3) -ll:', round(res_start.fun, 3))
    t_end = time.time()
    print(f'It took: {round(t_end - t_start, 3)} seconds.')

    # unpack Stage-3 pack
    rho1_0  = pars[0]
    rho2_0  = pars[1]
    theta_0 = pars[2]        # log-nu
    beta0   = pars[3: 4 + p]
    lam0    = pars[4 + p: 4 + p + cR]

    # Grid over phi, kappa (vector) for given scaling_code
    phi_star, kap0_vec, kappa_scalar = grid_phi_kappa_scalar_vec(
        ctx, pars, scaling_code
    )
    phi_0 = np.arctanh(phi_star)

    if start_mode == "standard":
        par0 = np.concatenate([
            [phi_0, rho1_0, rho2_0, theta_0],
            beta0,
            lam0.ravel(),
            kap0_vec.ravel()
        ])
        bounds = construct_bounds(p, cR, 'general', start_mode="standard")

    elif start_mode == "local_hess":
        kappa_h0  = 0.0
        delta_raw0 = 0.5
        gamma0     = 0.5
        par0 = np.concatenate([
            [phi_0, rho1_0, rho2_0, theta_0],
            beta0,
            lam0.ravel(),
            kap0_vec.ravel(),
            [kappa_h0, delta_raw0, gamma0]
        ])
        bounds = construct_bounds(p, cR, 'general', start_mode="local_hess")

    elif start_mode == "ewma":
        delta_raw0, gamma0 = 1.0, 0.5
        par0 = np.concatenate([
            [phi_0, rho1_0, rho2_0, theta_0],
            beta0,
            lam0.ravel(),
            kap0_vec.ravel(),
            [delta_raw0, gamma0]
        ])
        bounds = construct_bounds(p, cR, 'general', start_mode="ewma")

    elif start_mode == "inv_hessian":
        delta0 = 0.0
        par0 = np.concatenate([
            [phi_0, rho1_0, rho2_0, theta_0],
            beta0,
            lam0.ravel(),
            kap0_vec.ravel(),
            [delta0]
        ])
        bounds = construct_bounds(p, cR, 'general', start_mode="inv_hessian")

    else:
        raise ValueError(f"Unknown start_mode '{start_mode}'")

    return par0, bounds
