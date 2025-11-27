# implicit_filter.py
#
# Unified implicit (Broyden-based) score-driven filters for multiple "start modes":
#   - 'standard'    : standard scalings (gasp, invFisher, sqrtInvFisher, identity)
#   - 'local_hess'  : local Hessian filter with [kappa_h, delta, gamma]
#   - 'ewma'        : EWMA-of-Hessian filter with [delta, gamma]
#   - 'inv_hessian' : inverse-Hessian-based filter with [delta]
#
# It is structurally parallel to explicit_filter.py, but uses an *implicit*
# update step solved with Broyden's method.

import time
import numpy as np

import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jax import lax
from jaxopt import ScipyBoundedMinimize, Broyden

jax.config.update("jax_enable_x64", True)

from determine_start_parameters import (
    build_start_params as build_start_params_startdet,
    StartDetContext,
    ScalingCode,
    _SCALING_CODE_MAP,
)

# =============================================================================
# Shared utilities
# =============================================================================

def _t_const(nu, R, ld1, ld2, lam_sum):
    """Student-t normalization term with spatial transforms."""
    return (
        jsp.special.gammaln((nu + R) / 2.0)
        - jsp.special.gammaln(nu / 2.0)
        - 0.5 * R * jnp.log(nu)
        - 0.5 * R * jnp.log(jnp.pi)
        + ld1 + ld2 - lam_sum
    )


def _bounds_to_box(bounds):
    """Convert list of (lb, ub) into separate numpy arrays."""
    lb = np.array([b[0] for b in bounds], dtype=np.float64)
    ub = np.array([b[1] for b in bounds], dtype=np.float64)
    return lb, ub


def _s_grad_standard(dev, Z2, lam_inv, nu, scaling_code_int: int):
    """
    Score contribution S(mu) for standard scalings:
        0: GASP
        1: inverse Fisher
        2: sqrt inverse Fisher
        3: "identity"/gradient
    """
    R = dev.shape[0]
    x = Z2 @ dev
    q = jnp.sum((x.squeeze(-1) ** 2) * lam_inv.squeeze(-1))
    alphat = 1.0 + q / nu

    Av = Z2.T @ (lam_inv * x)   # (R,1)

    S_gasp = x / alphat
    S_invF = ((nu + R + 2.0) / (nu * alphat)) * dev
    S_sqrt = (
        jnp.sqrt((nu + R + 2.0) / (nu + R))
        * (nu + R) / (nu * alphat)
    ) * (jnp.sqrt(lam_inv) * x)
    S_iden = ((nu + R) / (nu * alphat)) * Av

    code = jnp.array(scaling_code_int, dtype=jnp.int32)
    S = jnp.where(code == 0, S_gasp,
         jnp.where(code == 1, S_invF,
           jnp.where(code == 2, S_sqrt, S_iden)))
    return S, alphat


def _score_mu_and_alpha(dev, Z2, lam_inv, nu):
    """
    Score wrt the location state mu_t and Student-t alpha_t.
    Returns:
        g      : (R,1) score wrt mu_t
        alphat : scalar, 1 + q/nu
        Av     : (R,1) auxiliary term Z2' Λ^{-1} Z2 dev
    """
    x = Z2 @ dev  # (R,1)
    q = jnp.sum((x.squeeze(-1) ** 2) * lam_inv.squeeze(-1))
    alphat = 1.0 + q / nu
    Av = Z2.T @ (lam_inv * x)  # (R,1)
    R = dev.shape[0]
    g = ((nu + R) / nu) * (Av / alphat)  # (R,1)
    return g, alphat, Av


def spectral_floor_and_expm(H, delta, eps=1e-8):
    """
    Spectral flooring of H to SPD, and exp(-delta * H_spd).
    Returns:
        H_spd : symmetrized, eigenvalue-floored version of H
        ew    : matrix exponential exp(-delta H_spd)
    """
    Hsym = 0.5 * (H + H.T)
    evals, evecs = jnp.linalg.eigh(Hsym)
    evals_clip = jnp.minimum(jnp.maximum(evals, eps), 1e6)  # floor + mild cap
    H_spd = (evecs * evals_clip) @ evecs.T
    ew = (evecs * jnp.exp(-delta * evals_clip)) @ evecs.T
    return H_spd, ew


# =============================================================================
# NLL implementations (per start_mode)
# =============================================================================

def implicit_nll_standard(parameters, ctx: StartDetContext, scaling_code: ScalingCode):
    """
    NLL for implicit filter with standard scalings (GASP, invF, sqrtInvF, identity),
    solved via Broyden at each time step.
    """
    Y, X, W1, W2 = ctx.Y, ctx.X, ctx.W1, ctx.W2
    Y = jnp.asarray(Y); X = jnp.asarray(X)
    W1 = jnp.asarray(W1); W2 = jnp.asarray(W2)

    T, R = Y.shape
    p = ctx.p

    # Parameters (same layout as explicit_filter / start_det)
    phi   = jnp.tanh(parameters[0])
    rho1  = jnp.tanh(parameters[1])
    rho2  = jnp.tanh(parameters[2])
    nu    = 2.0 + jnp.exp(parameters[3])
    beta  = parameters[4 : 5 + p].reshape(p + 1, 1)
    lam   = parameters[5 + p : 5 + p + R].reshape(R, 1)
    kappa = parameters[5 + p + R : 5 + p + 2 * R].reshape(R, 1)

    lam_inv = jnp.exp(-2.0 * lam)

    I = jnp.eye(R)
    Z1 = I - rho1 * W1
    Z2 = I - rho2 * W2

    _, ld1 = jnp.linalg.slogdet(Z1)
    _, ld2 = jnp.linalg.slogdet(Z2)

    const = _t_const(nu, R, ld1, ld2, lam.sum())
    scaling_code_int = int(scaling_code.value)

    # residual mapping for Broyden
    def F_fun(mu, pars):
        Z1y, xbeta, mu_prior, Z2_p, lam_inv_p, nu_p, kappa_p, sc = pars
        dev = Z1y - xbeta - mu
        S, _ = _s_grad_standard(dev, Z2_p, lam_inv_p, nu_p, sc)
        return mu - mu_prior - kappa_p * S

    solver = Broyden(fun=F_fun, tol=1e-8, maxiter=300, implicit_diff=True)

    def step(mu_prev, t):
        y_t   = Y[t].reshape(R, 1)
        xbeta = (X[t] @ beta).reshape(R, 1)
        Z1y   = Z1 @ y_t

        # log-likelihood uses PRIOR (predictive density)
        dev_prior = Z1y - xbeta - mu_prev
        _, a_t = _s_grad_standard(dev_prior, Z2, lam_inv, nu, scaling_code_int)
        ll_t = const - 0.5 * (nu + R) * jnp.log(a_t)

        pars = (Z1y, xbeta, mu_prev, Z2, lam_inv, nu, kappa, scaling_code_int)
        mu_t, _ = solver.run(mu_prev, pars)
        mu_next = phi * mu_t

        return mu_next, ll_t

    mu0 = jnp.zeros((R, 1))
    _, ll_seq = lax.scan(step, mu0, jnp.arange(T))
    return -jnp.sum(ll_seq)


def implicit_nll_ewma(parameters, ctx: StartDetContext, scaling_code: ScalingCode):
    """
    NLL for implicit EWMA-of-Hessian filter (delta, gamma).
    scaling_code is accepted for interface symmetry but not used.
    """
    del scaling_code  # unused, kept for API symmetry

    Y, X, W1, W2 = ctx.Y, ctx.X, ctx.W1, ctx.W2
    Y = jnp.asarray(Y); X = jnp.asarray(X)
    W1 = jnp.asarray(W1); W2 = jnp.asarray(W2)

    T, R = Y.shape
    p = ctx.p

    phi   = jnp.tanh(parameters[0])
    rho1  = jnp.tanh(parameters[1])
    rho2  = jnp.tanh(parameters[2])
    nu    = 2.0 + jnp.exp(parameters[3])
    beta  = parameters[4 : 5 + p].reshape(p + 1, 1)
    lam   = parameters[5 + p : 5 + p + R].reshape(R, 1)
    kappa = parameters[5 + p + R : 5 + p + 2 * R].reshape(R, 1)
    delta = parameters[5 + p + 2 * R]
    gamma = parameters[5 + p + 2 * R + 1]

    lam_inv = jnp.exp(-2.0 * lam)

    I = jnp.eye(R)
    Z1 = I - rho1 * W1
    Z2 = I - rho2 * W2

    _, ld1 = jnp.linalg.slogdet(Z1)
    _, ld2 = jnp.linalg.slogdet(Z2)

    const = _t_const(nu, R, ld1, ld2, lam.sum())

    Dlam = jnp.diag(lam_inv.reshape(-1))
    A = Z2.T @ Dlam @ Z2  # SPD baseline

    def ll_from_alph(alph):
        return const - 0.5 * (nu + R) * jnp.log(alph)

    def F_fun(mu, pars):
        Z1y, xbeta, Z2_p, nu_p, lam_inv_p, Ht, e, mu_prior, kappa_p = pars
        dev = Z1y - xbeta - mu
        g, _, _ = _score_mu_and_alpha(dev, Z2_p, lam_inv_p, nu_p)
        g = g.reshape(-1)
        step = (kappa_p.reshape(-1) * (e @ g)).reshape(R, 1)
        return mu - mu_prior - step

    solver = Broyden(fun=F_fun, tol=1e-6, maxiter=100, implicit_diff=True)

    def step(carry, t):
        mu_prev, Htilde_t = carry

        y_t   = Y[t].reshape(R, 1)
        xbeta = (X[t] @ beta).reshape(R, 1)
        Z1y   = Z1 @ y_t

        # Log-lik at prior
        dev = Z1y - xbeta - mu_prev
        _, alph, Av = _score_mu_and_alpha(dev, Z2, lam_inv, nu)
        ll_t = ll_from_alph(alph)

        # Instantaneous Hessian
        cfac = (nu + R) / nu
        H = (cfac / alph) * A - (2.0 * cfac / (nu * alph * alph)) * (Av @ Av.T)
        H = 0.5 * (H + H.T)

        # EWMA smoothing
        Htilde_next = gamma * Htilde_t + (1.0 - gamma) * H
        Htilde_t, ew = spectral_floor_and_expm(Htilde_next, delta, 1e-8)

        pars = (Z1y, xbeta, Z2, nu, lam_inv, Htilde_t, ew, mu_prev, kappa)
        mu_t, _ = solver.run(mu_prev, pars)
        mu_next = phi * mu_t

        return (mu_next, Htilde_t), ll_t

    mu0 = jnp.zeros((R, 1))
    Htilde0 = 0.0 * jnp.eye(R)
    (_, ll_seq) = lax.scan(step, (mu0, Htilde0), jnp.arange(T))
    return -jnp.sum(ll_seq)


def implicit_nll_local_hess(parameters, ctx: StartDetContext, scaling_code: ScalingCode):
    """
    NLL for implicit local-Hessian filter (kappa_h, delta, gamma).
    scaling_code is accepted for interface symmetry but not used.
    """
    del scaling_code

    Y, X, W1, W2 = ctx.Y, ctx.X, ctx.W1, ctx.W2
    Y = jnp.asarray(Y); X = jnp.asarray(X)
    W1 = jnp.asarray(W1); W2 = jnp.asarray(W2)

    T, R = Y.shape
    p = ctx.p

    phi   = jnp.tanh(parameters[0])
    rho1  = jnp.tanh(parameters[1])
    rho2  = jnp.tanh(parameters[2])
    nu    = 2.0 + jnp.exp(parameters[3])
    beta  = parameters[4 : 5 + p].reshape(p + 1, 1)
    lam   = parameters[5 + p : 5 + p + R].reshape(R, 1)
    kappa = parameters[5 + p + R : 5 + p + 2 * R].reshape(R, 1)

    kappa_h = parameters[5 + p + 2 * R]
    delta   = parameters[5 + p + 2 * R + 1]
    gamma   = parameters[5 + p + 2 * R + 2]

    lam_inv = jnp.exp(-2.0 * lam)

    I = jnp.eye(R)
    Z1 = I - rho1 * W1
    Z2 = I - rho2 * W2

    _, ld1 = jnp.linalg.slogdet(Z1)
    _, ld2 = jnp.linalg.slogdet(Z2)

    const = _t_const(nu, R, ld1, ld2, lam.sum())

    Dlam = jnp.diag(lam_inv.reshape(-1))
    A = Z2.T @ Dlam @ Z2  # SPD baseline

    def ll_from_alph(alph):
        return const - 0.5 * (nu + R) * jnp.log(alph)

    def F_fun(mu, pars):
        Z1y, xbeta, Z2_p, nu_p, lam_inv_p, Ht, e, mu_prior, kappa_p = pars
        dev = Z1y - xbeta - mu
        g, _, _ = _score_mu_and_alpha(dev, Z2_p, lam_inv_p, nu_p)
        g = g.reshape(-1)
        step = (kappa_p.reshape(-1) * (e @ g)).reshape(R, 1)
        return mu - mu_prior - step

    solver = Broyden(fun=F_fun, tol=1e-6, maxiter=200, implicit_diff=True)

    def step(carry, t):
        mu_prev, Htilde_t = carry

        y_t   = Y[t].reshape(R, 1)
        xbeta = (X[t] @ beta).reshape(R, 1)
        Z1y   = Z1 @ y_t

        dev = Z1y - xbeta - mu_prev
        _, alph, Av = _score_mu_and_alpha(dev, Z2, lam_inv, nu)
        ll_t = ll_from_alph(alph)

        cfac = (nu + R) / nu
        H = (cfac / alph) * A - (2.0 * cfac / (nu * alph * alph)) * (Av @ Av.T)
        H = 0.5 * (H + H.T)

        Htilde_next = kappa_h * jnp.eye(R, dtype=H.dtype) + gamma * Htilde_t + H
        Htilde_t, ew = spectral_floor_and_expm(Htilde_next, delta, 1e-8)

        pars = (Z1y, xbeta, Z2, nu, lam_inv, Htilde_t, ew, mu_prev, kappa)
        mu_tt, _ = solver.run(mu_prev, pars)
        mu_next = phi * mu_tt

        return (mu_next, Htilde_t), ll_t

    mu0 = jnp.zeros((R, 1))
    Htilde0 = 0.0 * jnp.eye(R)
    (_, ll_seq) = lax.scan(step, (mu0, Htilde0), jnp.arange(T))
    return -jnp.sum(ll_seq)


def implicit_nll_inv_hessian(parameters, ctx: StartDetContext, scaling_code: ScalingCode):
    """
    NLL for implicit inverse-Hessian filter (delta).
    scaling_code is accepted for interface symmetry but not used.
    """
    del scaling_code

    Y, X, W1, W2 = ctx.Y, ctx.X, ctx.W1, ctx.W2
    Y = jnp.asarray(Y); X = jnp.asarray(X)
    W1 = jnp.asarray(W1); W2 = jnp.asarray(W2)

    T, R = Y.shape
    p = ctx.p

    phi   = jnp.tanh(parameters[0])
    rho1  = jnp.tanh(parameters[1])
    rho2  = jnp.tanh(parameters[2])
    nu    = 2.0 + jnp.exp(parameters[3])
    beta  = parameters[4 : 5 + p].reshape(p + 1, 1)
    lam   = parameters[5 + p : 5 + p + R].reshape(R, 1)
    kappa = parameters[5 + p + R : 5 + p + 2 * R].reshape(R, 1)
    delta = parameters[5 + p + 2 * R]

    lam_inv = jnp.exp(-2.0 * lam)

    I = jnp.eye(R)
    Z1 = I - rho1 * W1
    Z2 = I - rho2 * W2

    _, ld1 = jnp.linalg.slogdet(Z1)
    _, ld2 = jnp.linalg.slogdet(Z2)

    const = _t_const(nu, R, ld1, ld2, lam.sum())

    Dlam = jnp.diag(lam_inv.reshape(-1))
    A = Z2.T @ Dlam @ Z2  # SPD baseline

    def ll_from_alph(alph):
        return const - 0.5 * (nu + R) * jnp.log(alph)

    def F_fun(mu, pars):
        Z1y, xbeta, mu_prior, A_p, Z2_p, lam_inv_p, nu_p, kappa_p, Ht, e = pars
        dev = Z1y - xbeta - mu

        x = Z2_p @ dev
        q = jnp.sum((x.squeeze(-1) ** 2) * lam_inv_p.squeeze(-1))
        Av = Z2_p.T @ (lam_inv_p * x)
        alphat = 1.0 + q / nu_p

        Rloc = dev.shape[0]
        cfac = (nu_p + Rloc) / nu_p
        g = cfac * (Av / alphat).reshape(-1)  # (R,)

        step = (kappa_p.reshape(-1) * (e @ g)).reshape(Rloc, 1)
        return mu - mu_prior - step

    solver = Broyden(fun=F_fun, tol=1e-6, maxiter=100, implicit_diff=True)

    def step(carry, t):
        mu_prev, H_tilde_prev = carry

        y_t   = Y[t].reshape(R, 1)
        xbeta = (X[t] @ beta).reshape(R, 1)
        Z1y   = Z1 @ y_t

        dev_prior = Z1y - xbeta - mu_prev
        _, a_t, _ = _score_mu_and_alpha(dev_prior, Z2, lam_inv, nu)
        ll_t = ll_from_alph(a_t)

        H_tilde_next, ew = spectral_floor_and_expm(H_tilde_prev, delta, 1e-8)

        pars = (Z1y, xbeta, mu_prev, A, Z2, lam_inv, nu, kappa, H_tilde_next, ew)
        mu_t, _ = solver.run(mu_prev, pars)
        mu_next = phi * mu_t

        return (mu_next, H_tilde_next), ll_t

    mu0 = jnp.zeros((R, 1))
    Htilde0 = 0.0 * jnp.eye(R)
    (_, ll_seq) = lax.scan(step, (mu0, Htilde0), jnp.arange(T))
    return -jnp.sum(ll_seq)


# -----------------------------------------------------------------------------
# Dispatcher + optimizer wrapper
# -----------------------------------------------------------------------------

def implicit_nll(parameters, ctx: StartDetContext, scaling_code: ScalingCode, start_mode: str):
    """Dispatch to the appropriate NLL based on start_mode."""
    if start_mode == "standard":
        return implicit_nll_standard(parameters, ctx, scaling_code)
    elif start_mode == "ewma":
        return implicit_nll_ewma(parameters, ctx, scaling_code)
    elif start_mode == "local_hess":
        return implicit_nll_local_hess(parameters, ctx, scaling_code)
    elif start_mode == "inv_hessian":
        return implicit_nll_inv_hessian(parameters, ctx, scaling_code)
    else:
        raise ValueError(f"Unknown start_mode='{start_mode}'")


def implicit_fit_jax(
    ctx: StartDetContext,
    scaling_code: ScalingCode,
    start_mode: str,
    par0,
    bounds,
    maxiter: int = 1500,
):
    """
    L-BFGS-B fit for the implicit filter, dispatching on start_mode.
    """
    fun = lambda th: implicit_nll(th, ctx, scaling_code, start_mode)

    lb, ub = _bounds_to_box(bounds)
    lb_j = jnp.asarray(lb, dtype=jnp.float64)
    ub_j = jnp.asarray(ub, dtype=jnp.float64)

    solver = ScipyBoundedMinimize(
        method="l-bfgs-b",
        fun=fun,
        jit=True,
        maxiter=maxiter,
        options={"ftol": 1e-12, "gtol": 1e-10},
    )

    x0 = jnp.asarray(par0, dtype=jnp.float64)
    params_opt, info = solver.run(x0, (lb_j, ub_j))
    return params_opt, info


def _resolve_scaling_code(scaling):
    """
    Map 'gasp', 'invFisher', 'sqrtInvFisher', 'identity' or ScalingCode -> ScalingCode.
    """
    if isinstance(scaling, ScalingCode):
        return scaling
    if isinstance(scaling, str):
        try:
            return _SCALING_CODE_MAP[scaling]
        except KeyError:
            raise ValueError(
                f"Unknown scaling='{scaling}'. "
                f"Expected one of {list(_SCALING_CODE_MAP.keys())}"
            )
    raise TypeError("scaling must be a string or ScalingCode")


# =============================================================================
# Evaluation (per start_mode) – parallel to explicit_filter.py
# =============================================================================

def implicit_eval_standard(parameters, ctx: StartDetContext, scaling_code: ScalingCode, mu0):
    """Evaluation roll for standard-scaling implicit filter."""
    Y, X, W1, W2 = ctx.Y, ctx.X, ctx.W1, ctx.W2
    Y = jnp.asarray(Y); X = jnp.asarray(X)
    W1 = jnp.asarray(W1); W2 = jnp.asarray(W2)

    T, R = Y.shape
    p = ctx.p

    phi   = jnp.tanh(parameters[0])
    rho1  = jnp.tanh(parameters[1])
    rho2  = jnp.tanh(parameters[2])
    nu    = 2.0 + jnp.exp(parameters[3])
    beta  = parameters[4 : 5 + p].reshape(p + 1, 1)
    lam   = parameters[5 + p : 5 + p + R].reshape(R, 1)
    kappa = parameters[5 + p + R : 5 + p + 2 * R].reshape(R, 1)

    lam_inv = jnp.exp(-2.0 * lam)

    I = jnp.eye(R)
    Z1 = I - rho1 * W1
    Z2 = I - rho2 * W2

    _, ld1 = jnp.linalg.slogdet(Z1)
    _, ld2 = jnp.linalg.slogdet(Z2)
    const = _t_const(nu, R, ld1, ld2, lam.sum())

    scaling_code_int = int(scaling_code.value)

    def ll_from_alph(alph):
        return const - 0.5 * (nu + R) * jnp.log(alph)

    def F_fun(mu, pars):
        Z1y, xbeta, mu_prior, Z2_p, lam_inv_p, nu_p, kappa_p, sc = pars
        dev = Z1y - xbeta - mu
        S, _ = _s_grad_standard(dev, Z2_p, lam_inv_p, nu_p, sc)
        return mu - mu_prior - kappa_p * S

    solver = Broyden(fun=F_fun, tol=1e-8, maxiter=200, implicit_diff=False)

    def step(mu_prev, t):
        y_t   = Y[t].reshape(R, 1)
        xbeta = (X[t] @ beta).reshape(R, 1)
        Z1y   = Z1 @ y_t

        yhat_t = jnp.linalg.solve(Z1, xbeta + mu_prev).reshape(-1)

        dev_prior = Z1y - xbeta - mu_prev
        _, a_t = _s_grad_standard(dev_prior, Z2, lam_inv, nu, scaling_code_int)
        ll_t = ll_from_alph(a_t)

        pars = (Z1y, xbeta, mu_prev, Z2, lam_inv, nu, kappa, scaling_code_int)
        mu_t, _ = solver.run(mu_prev, pars)
        mu_next = phi * mu_t

        return mu_next, (mu_prev.reshape(-1), ll_t, yhat_t, mu_t)

    (mu_pred_last, (mu_seq_flat, ll_seq, yhat_seq, mu_filt_seq)) = lax.scan(
        step, mu0, jnp.arange(T)
    )

    mu_seq = mu_seq_flat.reshape(T, R)
    yhat = yhat_seq.reshape(T, R)
    mu_filt = mu_filt_seq.reshape(T, R)

    return mu_seq, ll_seq, yhat, mu_pred_last, mu_filt


def implicit_eval_ewma(parameters, ctx: StartDetContext, mu0, Htilde0):
    """Evaluation roll for implicit EWMA-of-Hessian filter."""
    Y, X, W1, W2 = ctx.Y, ctx.X, ctx.W1, ctx.W2
    Y = jnp.asarray(Y); X = jnp.asarray(X)
    W1 = jnp.asarray(W1); W2 = jnp.asarray(W2)

    T, R = Y.shape
    p = ctx.p

    phi   = jnp.tanh(parameters[0])
    rho1  = jnp.tanh(parameters[1])
    rho2  = jnp.tanh(parameters[2])
    nu    = 2.0 + jnp.exp(parameters[3])
    beta  = parameters[4 : 5 + p].reshape(p + 1, 1)
    lam   = parameters[5 + p : 5 + p + R].reshape(R, 1)
    kappa = parameters[5 + p + R : 5 + p + 2 * R].reshape(R, 1)

    delta = parameters[5 + p + 2 * R]
    gamma = parameters[5 + p + 2 * R + 1]

    lam_inv = jnp.exp(-2.0 * lam)

    I = jnp.eye(R)
    Z1 = I - rho1 * W1
    Z2 = I - rho2 * W2

    _, ld1 = jnp.linalg.slogdet(Z1)
    _, ld2 = jnp.linalg.slogdet(Z2)
    const = _t_const(nu, R, ld1, ld2, lam.sum())

    Dlam = jnp.diag(lam_inv.reshape(-1))
    A = Z2.T @ Dlam @ Z2

    def ll_from_alph(alph):
        return const - 0.5 * (nu + R) * jnp.log(alph)

    def F_fun(mu, pars):
        Z1y, xbeta, Z2_p, nu_p, lam_inv_p, Ht, e, mu_prior, kappa_p = pars
        dev = Z1y - xbeta - mu
        g, _, _ = _score_mu_and_alpha(dev, Z2_p, lam_inv_p, nu_p)
        g = g.reshape(-1)
        step = (kappa_p.reshape(-1) * (e @ g)).reshape(R, 1)
        return mu - mu_prior - step

    solver = Broyden(fun=F_fun, tol=1e-8, maxiter=100, implicit_diff=False)

    def step(carry, t):
        mu_prev, Htilde_t = carry

        y_t   = Y[t].reshape(R, 1)
        xbeta = (X[t] @ beta).reshape(R, 1)
        Z1y   = Z1 @ y_t

        yhat_t = jnp.linalg.solve(Z1, xbeta + mu_prev).reshape(-1)

        dev = Z1y - xbeta - mu_prev
        g_t, H_alph, Av = _score_mu_and_alpha(dev, Z2, lam_inv, nu)
        ll_t = ll_from_alph(H_alph)

        cfac = (nu + R) / nu
        H = (cfac / H_alph) * A - (2.0 * cfac / (nu * H_alph * H_alph)) * (Av @ Av.T)
        H = 0.5 * (H + H.T)

        Htilde_next = gamma * Htilde_t + (1.0 - gamma) * H
        Htilde_t, ew = spectral_floor_and_expm(Htilde_next, delta, 1e-8)

        pars = (Z1y, xbeta, Z2, nu, lam_inv, Htilde_t, ew, mu_prev, kappa)
        mu_tt, _ = solver.run(mu_prev, pars)
        mu_next = phi * mu_tt

        return (mu_next, Htilde_t), (mu_prev.reshape(-1), ll_t, yhat_t, mu_tt)

    (carry_last, (mu_seq_flat, ll_seq, yhat_seq, mu_filt_seq)) = lax.scan(
        step, (mu0, Htilde0), jnp.arange(T)
    )
    mu_pred_last, Htilde_last = carry_last

    mu_seq = mu_seq_flat.reshape(T, R)
    yhat = yhat_seq.reshape(T, R)
    mu_filt = mu_filt_seq.reshape(T, R)

    return mu_seq, ll_seq, yhat, mu_pred_last, Htilde_last, mu_filt


def implicit_eval_local_hess(parameters, ctx: StartDetContext, mu0, Htilde0):
    """Evaluation roll for implicit local-Hessian filter."""
    Y, X, W1, W2 = ctx.Y, ctx.X, ctx.W1, ctx.W2
    Y = jnp.asarray(Y); X = jnp.asarray(X)
    W1 = jnp.asarray(W1); W2 = jnp.asarray(W2)

    T, R = Y.shape
    p = ctx.p

    phi   = jnp.tanh(parameters[0])
    rho1  = jnp.tanh(parameters[1])
    rho2  = jnp.tanh(parameters[2])
    nu    = 2.0 + jnp.exp(parameters[3])
    beta  = parameters[4 : 5 + p].reshape(p + 1, 1)
    lam   = parameters[5 + p : 5 + p + R].reshape(R, 1)
    kappa = parameters[5 + p + R : 5 + p + 2 * R].reshape(R, 1)

    kappa_h = parameters[5 + p + 2 * R]
    delta   = parameters[5 + p + 2 * R + 1]
    gamma   = parameters[5 + p + 2 * R + 2]

    lam_inv = jnp.exp(-2.0 * lam)

    I = jnp.eye(R)
    Z1 = I - rho1 * W1
    Z2 = I - rho2 * W2

    _, ld1 = jnp.linalg.slogdet(Z1)
    _, ld2 = jnp.linalg.slogdet(Z2)
    const = _t_const(nu, R, ld1, ld2, lam.sum())

    Dlam = jnp.diag(lam_inv.reshape(-1))
    A = Z2.T @ Dlam @ Z2

    def ll_from_alph(alph):
        return const - 0.5 * (nu + R) * jnp.log(alph)

    def F_fun(mu, pars):
        Z1y, xbeta, Z2_p, nu_p, lam_inv_p, Ht, e, mu_prior, kappa_p = pars
        dev = Z1y - xbeta - mu
        g, _, _ = _score_mu_and_alpha(dev, Z2_p, lam_inv_p, nu_p)
        g = g.reshape(-1)
        step = (kappa_p.reshape(-1) * (e @ g)).reshape(R, 1)
        return mu - mu_prior - step

    solver = Broyden(fun=F_fun, tol=1e-8, maxiter=200, implicit_diff=False)

    def step(carry, t):
        mu_prev, Htilde_t = carry

        y_t   = Y[t].reshape(R, 1)
        xbeta = (X[t] @ beta).reshape(R, 1)
        Z1y   = Z1 @ y_t

        yhat_t = jnp.linalg.solve(Z1, xbeta + mu_prev).reshape(-1)

        dev = Z1y - xbeta - mu_prev
        g_t, H_alph, Av = _score_mu_and_alpha(dev, Z2, lam_inv, nu)
        ll_t = ll_from_alph(H_alph)

        cfac = (nu + R) / nu
        H = (cfac / H_alph) * A - (2.0 * cfac / (nu * H_alph * H_alph)) * (Av @ Av.T)
        H = 0.5 * (H + H.T)

        Htilde_next = kappa_h * jnp.eye(R, dtype=H.dtype) + gamma * Htilde_t + H
        Htilde_t, ew = spectral_floor_and_expm(Htilde_next, delta, 1e-8)

        pars = (Z1y, xbeta, Z2, nu, lam_inv, Htilde_t, ew, mu_prev, kappa)
        mu_tt, _ = solver.run(mu_prev, pars)
        mu_next = phi * mu_tt

        return (mu_next, Htilde_t), (mu_prev.reshape(-1), ll_t, yhat_t, mu_tt)

    (carry_last, (mu_seq_flat, ll_seq, yhat_seq, mu_filt_seq)) = lax.scan(
        step, (mu0, Htilde0), jnp.arange(T)
    )
    mu_pred_last, Htilde_last = carry_last

    mu_seq = mu_seq_flat.reshape(T, R)
    yhat = yhat_seq.reshape(T, R)
    mu_filt = mu_filt_seq.reshape(T, R)

    return mu_seq, ll_seq, yhat, mu_pred_last, Htilde_last, mu_filt


def implicit_eval_inv_hessian(parameters, ctx: StartDetContext, mu0, Htilde0):
    """Evaluation roll for implicit inverse-Hessian filter."""
    Y, X, W1, W2 = ctx.Y, ctx.X, ctx.W1, ctx.W2
    Y = jnp.asarray(Y); X = jnp.asarray(X)
    W1 = jnp.asarray(W1); W2 = jnp.asarray(W2)

    T, R = Y.shape
    p = ctx.p

    phi   = jnp.tanh(parameters[0])
    rho1  = jnp.tanh(parameters[1])
    rho2  = jnp.tanh(parameters[2])
    nu    = 2.0 + jnp.exp(parameters[3])
    beta  = parameters[4 : 5 + p].reshape(p + 1, 1)
    lam   = parameters[5 + p : 5 + p + R].reshape(R, 1)
    kappa = parameters[5 + p + R : 5 + p + 2 * R].reshape(R, 1)
    delta = parameters[5 + p + 2 * R]

    lam_inv = jnp.exp(-2.0 * lam)

    I = jnp.eye(R)
    Z1 = I - rho1 * W1
    Z2 = I - rho2 * W2

    _, ld1 = jnp.linalg.slogdet(Z1)
    _, ld2 = jnp.linalg.slogdet(Z2)
    const = _t_const(nu, R, ld1, ld2, lam.sum())

    Dlam = jnp.diag(lam_inv.reshape(-1))
    A = Z2.T @ Dlam @ Z2

    def ll_from_alph(alph):
        return const - 0.5 * (nu + R) * jnp.log(alph)

    def F_fun(mu, pars):
        Z1y, xbeta, mu_prior, A_p, Z2_p, lam_inv_p, nu_p, kappa_p, Ht, e = pars
        dev = Z1y - xbeta - mu
        x = Z2_p @ dev
        q = jnp.sum((x.squeeze(-1) ** 2) * lam_inv_p.squeeze(-1))
        Av = Z2_p.T @ (lam_inv_p * x)
        alph = 1.0 + q / nu_p
        Rloc = dev.shape[0]
        cfac = (nu_p + Rloc) / nu_p
        g = cfac * (Av / alph).reshape(-1)
        step = (kappa_p.reshape(-1) * (e @ g)).reshape(Rloc, 1)
        return mu - mu_prior - step

    solver = Broyden(fun=F_fun, tol=1e-8, maxiter=200, implicit_diff=False)

    def step(carry, t):
        mu_prev, H_tilde_prev = carry

        y_t   = Y[t].reshape(R, 1)
        xbeta = (X[t] @ beta).reshape(R, 1)
        Z1y   = Z1 @ y_t

        y_pred_t = jnp.linalg.solve(Z1, xbeta + mu_prev).reshape(-1)

        dev_prior = Z1y - xbeta - mu_prev
        _, a_t, _ = _score_mu_and_alpha(dev_prior, Z2, lam_inv, nu)
        ll_t = ll_from_alph(a_t)

        H_tilde_next, ew = spectral_floor_and_expm(H_tilde_prev, delta, 1e-8)

        pars = (Z1y, xbeta, mu_prev, A, Z2, lam_inv, nu, kappa, H_tilde_next, ew)
        mu_t, _ = solver.run(mu_prev, pars)
        mu_next = phi * mu_t

        return (mu_next, H_tilde_next), (mu_prev.reshape(-1), ll_t, y_pred_t, mu_t)

    carry_last, (mu_seq_flat, ll_seq, yhat_seq, mu_filt_seq) = lax.scan(
        step, (mu0, Htilde0), jnp.arange(T)
    )
    mu_pred_last, Htilde_last = carry_last

    mu_seq = mu_seq_flat.reshape(T, R)
    yhat = yhat_seq.reshape(T, R)
    mu_filt = mu_filt_seq.reshape(T, R)

    return mu_seq, ll_seq, yhat, mu_pred_last, Htilde_last, mu_filt


# =============================================================================
# High-level helpers – fit + evaluate (in-sample / out-of-sample)
# =============================================================================

def fit_and_evaluate(
    Y,
    X,
    W,
    scaling,
    start_mode: str = "standard",
    save_str: str = "start_params.pkl",
    maxiter: int = 1500,
):
    """
    Fit implicit filter and compute in-sample performance metrics.

    Output structure matches explicit_filter.fit_and_evaluate.
    """
    Y = np.asarray(Y)
    X = np.asarray(X)
    W = np.asarray(W)

    cT, cR = Y.shape

    ctx = StartDetContext(Y=Y, X=X, W1=W, W2=W)
    scaling_code = _resolve_scaling_code(scaling)

    # Start parameters from the unified start-det routine
    par0_np, bounds = build_start_params_startdet(
        Y,
        X,
        W,
        W,
        cb=False,
        scaling=scaling,
        start=None,
        save_str=save_str,
        start_mode=start_mode,
    )
    par0 = jnp.asarray(par0_np, dtype=jnp.float64)

    # Fit
    t0 = time.time()
    print("--- JAX L-BFGS-B fit (implicit NLL) ---")
    params_opt, state = implicit_fit_jax(
        ctx, scaling_code, start_mode, par0, bounds, maxiter=maxiter
    )
    t1 = time.time()
    print("Finished JAX fit. Final NLL:", float(state.fun_val))
    print(f"It took: {round(t1 - t0, 4)} seconds.")

    # In-sample evaluation
    mu0 = jnp.zeros((cR, 1))

    if start_mode in ("ewma", "local_hess"):
        Htilde0 = 0.0 * jnp.eye(cR)
        if start_mode == "ewma":
            mu_in, llik_in, yhat_in, _, _, mu_filt_in = implicit_eval_ewma(
                params_opt, ctx, mu0, Htilde0
            )
        else:  # local_hess
            mu_in, llik_in, yhat_in, _, _, mu_filt_in = implicit_eval_local_hess(
                params_opt, ctx, mu0, Htilde0
            )
    elif start_mode == "standard":
        mu_in, llik_in, yhat_in, _, mu_filt_in = implicit_eval_standard(
            params_opt, ctx, scaling_code, mu0
        )
    elif start_mode == "inv_hessian":
        Htilde0 = 0.0 * jnp.eye(cR)
        mu_in, llik_in, yhat_in, _, _, mu_filt_in = implicit_eval_inv_hessian(
            params_opt, ctx, mu0, Htilde0
        )
    else:
        raise ValueError(f"Unknown start_mode '{start_mode}'")

    mu_in_np = np.array(mu_in)
    llik_in_np = np.array(llik_in)
    yhat_in_np = np.array(yhat_in)

    avg_log_in = float(llik_in_np.mean())
    print("--- Evaluation ---")
    print(f"In-sample avg log-score:  {avg_log_in:.8f}")

    return {
        "state": state,
        "params_opt": np.array(params_opt),
        "final_nll": float(state.fun_val),
        "mu_in": mu_in_np,
        "llik_in": llik_in_np,
        "yhat_in": yhat_in_np,
        "avg_log_in": avg_log_in,
        "bounds": bounds,
        "par0": np.array(par0_np),
    }


def implicit_analyse_oos(
    Y_in,
    X_in,
    mu_in,
    Y_out,
    X_out,
    mu_out,
    W,
    scaling,
    start_mode: str,
    cb_start: bool,
    cb_general: bool,
    save_str: str = "start_params.pkl",
    maxiter: int = 1500,
):
    """
    Fit on in-sample data and evaluate both in-sample and out-of-sample.

    Output structure matches explicit_filter.explicit_analyse_oos.
    """
    # Silence unused cb_general (kept for interface matching)
    _ = cb_general

    Y_in = np.asarray(Y_in)
    X_in = np.asarray(X_in)
    mu_in = None if mu_in is None else np.asarray(mu_in)

    Y_out = np.asarray(Y_out)
    X_out = np.asarray(X_out)
    mu_out = None if mu_out is None else np.asarray(mu_out)

    W = np.asarray(W)

    cT, cR = Y_in.shape

    ctx_in = StartDetContext(Y=Y_in, X=X_in, W1=W, W2=W)
    ctx_out = StartDetContext(Y=Y_out, X=X_out, W1=W, W2=W)

    scaling_code = _resolve_scaling_code(scaling)

    # Start params
    par0_np, bounds = build_start_params_startdet(
        Y_in,
        X_in,
        W,
        W,
        cb=cb_start,
        scaling=scaling,
        start=None,
        save_str=save_str,
        start_mode=start_mode,
    )
    par0 = jnp.asarray(par0_np, dtype=jnp.float64)

    # Fit
    t_start = time.time()
    print("--- JAX L-BFGS-B fit (implicit NLL, oos) ---")
    params_opt, state = implicit_fit_jax(
        ctx_in, scaling_code, start_mode, par0, bounds, maxiter=maxiter
    )
    t_end = time.time()
    print("Finished JAX fit. Final NLL:", float(state.fun_val))
    theta_opt = np.array(params_opt)
    print(f"It took: {round(t_end - t_start, 4)} seconds.")

    # In-sample eval
    mu0_in = jnp.zeros((cR, 1))

    if start_mode in ("ewma", "local_hess"):
        Htilde0 = 0.0 * jnp.eye(cR)
        if start_mode == "ewma":
            mu_in_hat, llik_in, y_in_hat, mu_pred_last, Htilde_last, mu_filt_in_hat = implicit_eval_ewma(
                params_opt, ctx_in, mu0_in, Htilde0
            )
        else:  # local_hess
            mu_in_hat, llik_in, y_in_hat, mu_pred_last, Htilde_last, mu_filt_in_hat = implicit_eval_local_hess(
                params_opt, ctx_in, mu0_in, Htilde0
            )
    elif start_mode == "standard":
        mu_in_hat, llik_in, y_in_hat, mu_pred_last, mu_filt_in_hat = implicit_eval_standard(
            params_opt, ctx_in, scaling_code, mu0_in
        )
        Htilde_last = None
    elif start_mode == "inv_hessian":
        Htilde0 = 0.0 * jnp.eye(cR)
        mu_in_hat, llik_in, y_in_hat, mu_pred_last, Htilde_last, mu_filt_in_hat = implicit_eval_inv_hessian(
            params_opt, ctx_in, mu0_in, Htilde0
        )
    else:
        raise ValueError(f"Unknown start_mode '{start_mode}'")

    mu_in_hat_np = np.array(mu_in_hat)
    llik_in_np = np.array(llik_in)
    y_in_hat_np = np.array(y_in_hat)
    mu_filt_in_hat_np = np.array(mu_filt_in_hat)

    avg_log_in = float(llik_in_np.mean())

    # In-sample error metrics
    err_y_in = np.asarray(Y_in) - y_in_hat_np
    mse_y_in = float(np.mean(err_y_in ** 2))
    mae_y_in = float(np.mean(np.abs(err_y_in)))

    if mu_in is not None:
        err_mu_in = np.asarray(mu_in) - mu_in_hat_np
        mse_mu_in = float(np.mean(err_mu_in ** 2))
        mae_mu_in = float(np.mean(np.abs(err_mu_in)))
    else:
        mse_mu_in = None
        mae_mu_in = None

    print("--- In-sample evaluation ---")
    print(f"In-sample avg log-score:  {avg_log_in:.8f}")

    # Out-of-sample eval
    mu0_out = mu_pred_last

    if start_mode in ("ewma", "local_hess"):
        if start_mode == "ewma":
            mu_out_hat, llik_out, y_out_hat, _, _, mu_filt_out_hat = implicit_eval_ewma(
                params_opt, ctx_out, mu0_out, Htilde_last
            )
        else:
            mu_out_hat, llik_out, y_out_hat, _, _, mu_filt_out_hat = implicit_eval_local_hess(
                params_opt, ctx_out, mu0_out, Htilde_last
            )
    elif start_mode == "standard":
        mu_out_hat, llik_out, y_out_hat, _, mu_filt_out_hat = implicit_eval_standard(
            params_opt, ctx_out, scaling_code, mu0_out
        )
    elif start_mode == "inv_hessian":
        Htilde0_out = Htilde_last if Htilde_last is not None else 0.0 * jnp.eye(cR)
        mu_out_hat, llik_out, y_out_hat, _, Htilde_last_out, mu_filt_out_hat = implicit_eval_inv_hessian(
            params_opt, ctx_out, mu0_out, Htilde0_out
        )
        _ = Htilde_last_out  # unused
    else:
        raise ValueError(f"Unknown start_mode '{start_mode}'")

    mu_out_hat_np = np.array(mu_out_hat)
    llik_out_np = np.array(llik_out)
    y_out_hat_np = np.array(y_out_hat)
    mu_filt_out_hat_np = np.array(mu_filt_out_hat)

    avg_log_out = float(llik_out_np.mean())

    # Out-of-sample error metrics
    err_y_out = np.asarray(Y_out) - y_out_hat_np
    mse_y_out = float(np.mean(err_y_out ** 2))
    mae_y_out = float(np.mean(np.abs(err_y_out)))

    if mu_out is not None:
        err_mu_out = np.asarray(mu_out) - mu_out_hat_np
        mse_mu_out = float(np.mean(err_mu_out ** 2))
        mae_mu_out = float(np.mean(np.abs(err_mu_out)))
    else:
        mse_mu_out = None
        mae_mu_out = None

    print("--- Out-of-sample evaluation ---")
    print(f"Out-sample avg log-score:  {avg_log_out:.8f}")

    return {
        "state": state,
        "opt": theta_opt,
        "params": np.array(params_opt),
        "-ll": float(state.fun_val),
        "mu_in_hat": mu_in_hat_np,
        "mu_out_hat": mu_out_hat_np,
        "mu_filt_in_hat": mu_filt_in_hat_np,
        "mu_filt_out_hat": mu_filt_out_hat_np,
        "llik_in": llik_in_np,
        "llik_out": llik_out_np,
        "y_hat_in": y_in_hat_np,
        "y_hat_out": y_out_hat_np,
        "metrics": {
            "avg_log_in": avg_log_in,
            "avg_log_out": avg_log_out,
            "mse_mu_in": mse_mu_in,
            "mse_mu_out": mse_mu_out,
            "mae_mu_in": mae_mu_in,
            "mae_mu_out": mae_mu_out,
            "mse_y_in": mse_y_in,
            "mse_y_out": mse_y_out,
            "mae_y_in": mae_y_in,
            "mae_y_out": mae_y_out,
        },
        "avg_log_in": avg_log_in,
        "bounds": bounds,
        "par0": np.array(par0_np),
    }
