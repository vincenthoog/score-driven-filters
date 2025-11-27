# explicit_filter.py
#
# Unified explicit score-driven filter implementations for multiple "start modes":
#   - 'standard'    : standard scalings (gasp, invFisher, sqrtInvFisher, identity)
#   - 'local_hess'  : local Hessian filter with [kappa_h, delta, gamma]
#   - 'ewma'        : EWMA-of-Hessian filter with [delta, gamma]
#   - 'inv_hessian' : inverse-Hessian-based filter with [delta]

import time
import numpy as np

import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jax import lax
from jaxopt import ScipyBoundedMinimize

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
    return (jsp.special.gammaln((nu + R) / 2.0)
            - jsp.special.gammaln(nu / 2.0)
            - 0.5 * R * jnp.log(nu)
            - 0.5 * R * jnp.log(jnp.pi)
            + ld1 + ld2 - lam_sum)


def _bounds_to_box(bounds):
    """Convert list of (lb, ub) into separate numpy arrays."""
    lb = np.array([b[0] for b in bounds], dtype=np.float64)
    ub = np.array([b[1] for b in bounds], dtype=np.float64)
    return lb, ub


# ---- Standard scalings: score S(mu) for different scaling_code --------------

def _s_grad_explicit(dev, Z2, lam_inv, nu, scaling_code_int: int):
    """
    Score contribution S(mu) for standard scalings:
        0: GASP
        1: inverse Fisher
        2: sqrt inverse Fisher
        3: "identity"/gradient
    Matches z_explicit_filter_standard_scalings.py.
    """
    R = dev.shape[0]
    x = Z2 @ dev
    q = jnp.sum((x.squeeze(-1)**2) * lam_inv.squeeze(-1))
    alphat = 1.0 + q / nu

    Av = Z2.T @ (lam_inv * x)   # (R,1)

    S_gasp = x / alphat
    S_invF = ((nu + R + 2.0) / (nu * alphat)) * dev
    S_sqrt = (jnp.sqrt((nu + R + 2.0) / (nu + R)) * (nu + R) / (nu * alphat)) * (jnp.sqrt(lam_inv) * x)
    S_iden = ((nu + R) / (nu * alphat)) * Av

    code = jnp.array(scaling_code_int, dtype=jnp.int32)
    S = jnp.where(code == 0, S_gasp,
         jnp.where(code == 1, S_invF,
           jnp.where(code == 2, S_sqrt, S_iden)))
    return S, alphat


# ---- Local Hessian / EWMA / inv-Hessian utilities ---------------------------

def _score_mu_and_alpha(dev, Z2, lam_inv, nu):
    """
    Score and alpha used in local-Hessian and EWMA variants.
    Matches the _score_mu_and_alpha definitions in original files.
    """
    x = Z2 @ dev
    q = jnp.sum((x.squeeze(-1)**2) * lam_inv.squeeze(-1))
    alphat = 1.0 + q / nu
    Av = Z2.T @ (lam_inv * x)
    R = dev.shape[0]
    g = ((nu + R) / nu) * (Av / alphat)
    return g.reshape(-1), alphat, Av


def spectral_floor_and_expm(H, delta, eps=1e-8):
    """
    Ensure H is SPD, then compute matrix exp(-delta * H).
    Used in EWMA and local-Hessian variants.
    """
    Hsym = 0.5 * (H + H.T)
    evals, evecs = jnp.linalg.eigh(Hsym)
    evals_clip = jnp.maximum(evals, eps)
    H_spd = (evecs * evals_clip) @ evecs.T
    ew = (evecs * jnp.exp(-delta * evals_clip)) @ evecs.T
    return H_spd, ew


def spectral_floor_and_expm_apply(H, v, delta, eps=1e-8):
    """
    Ensure H is SPD, then apply exp(-delta * H) to vector v.
    Used in inverse-Hessian variant.
    """
    Hsym = 0.5 * (H + H.T)
    evals, evecs = jnp.linalg.eigh(Hsym)
    evals = jnp.maximum(evals, eps)
    vh = evecs.T @ v
    w = jnp.exp(-delta * evals)[:, None]
    return evecs @ (w * vh)


def _s_grad_invHess(dev, A, Z2, lam_inv, nu, delta):
    """
    Inverse-Hessian style step:
      step = exp(-delta * H) * g,
    where H is the local Hessian and g is the score.
    Matches z_explicit_filter_invHessian.py.
    """
    x = Z2 @ dev
    q = jnp.sum((x.squeeze(-1)**2) * lam_inv.squeeze(-1))
    Av = Z2.T @ (lam_inv * x)
    alphat = 1.0 + q / nu
    R = dev.shape[0]
    cfac = (nu + R) / nu

    g = cfac * (Av / alphat)

    H = (cfac / alphat) * A - (2.0 * cfac / (nu * alphat * alphat)) * (Av @ Av.T)
    H = 0.5 * (H + H.T)

    step = spectral_floor_and_expm_apply(H, g, delta)
    return step, alphat


# =============================================================================
# Negative log-likelihoods per start_mode
# =============================================================================

def explicit_nll_standard(parameters, ctx: StartDetContext, scaling_code: ScalingCode):
    """NLL for standard scalings (no Hessian state)."""
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

    def step(mu_t, t):
        y_t   = Y[t].reshape(R, 1)
        xbeta = (X[t] @ beta).reshape(R, 1)
        dev   = (Z1 @ y_t) - xbeta - mu_t

        S, alph = _s_grad_explicit(dev, Z2, lam_inv, nu, scaling_code_int)
        ll_t = ll_from_alph(alph)

        mu_filt = mu_t + kappa * S
        mu_next = phi * mu_filt

        return mu_next, ll_t

    mu0 = jnp.zeros((R, 1))
    _, ll_seq = lax.scan(step, mu0, jnp.arange(T))
    total_ll = jnp.sum(ll_seq)
    return -total_ll


def explicit_nll_ewma(parameters, ctx: StartDetContext, scaling_code: ScalingCode):
    """NLL for EWMA-of-Hessian filter (delta, gamma)."""
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

    def step(carry, t):
        mu_prev, Htilde_t = carry

        y_t   = Y[t].reshape(R, 1)
        xbeta = (X[t] @ beta).reshape(R, 1)

        dev  = (Z1 @ y_t) - xbeta - mu_prev
        g_t, alph, Av = _score_mu_and_alpha(dev, Z2, lam_inv, nu)

        Htilde_t, ew = spectral_floor_and_expm(Htilde_t, delta)
        step_vec = (kappa.reshape(-1) * (ew @ g_t)).reshape(R, 1)
        mu_filt  = mu_prev + step_vec
        mu_t     = phi * mu_filt

        cfac = (nu + R) / nu
        H    = (cfac / alph) * A - (2.0 * cfac / (nu * alph * alph)) * (Av @ Av.T)
        H    = 0.5 * (H + H.T)

        Htilde_next = gamma * Htilde_t + (1.0 - gamma) * H

        return (mu_t, Htilde_next), ll_from_alph(alph)

    mu0 = jnp.zeros((R, 1))
    Htilde0 = 0.0 * jnp.eye(R)
    (_, ll_seq) = jax.lax.scan(step, (mu0, Htilde0), jnp.arange(T))
    total_ll = jnp.sum(ll_seq)
    return -total_ll


def explicit_nll_local_hess(parameters, ctx: StartDetContext, scaling_code: ScalingCode):
    """NLL for local-Hessian filter (kappa_h, delta, gamma)."""
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

    def step(carry, t):
        mu_prev, Htilde_t = carry

        y_t   = Y[t].reshape(R, 1)
        xbeta = (X[t] @ beta).reshape(R, 1)

        dev  = (Z1 @ y_t) - xbeta - mu_prev
        g_t, alph, Av = _score_mu_and_alpha(dev, Z2, lam_inv, nu)

        Htilde_t, ew = spectral_floor_and_expm(Htilde_t, delta)
        step_vec = (kappa.reshape(-1) * (ew @ g_t)).reshape(R, 1)

        mu_filt = mu_prev + step_vec
        mu_t = phi * mu_filt

        cfac = (nu + R) / nu
        H    = (cfac / alph) * A - (2.0 * cfac / (nu * alph * alph)) * (Av @ Av.T)
        H    = 0.5 * (H + H.T)

        Htilde_next = kappa_h * jnp.eye(R, dtype=H.dtype) + gamma * Htilde_t + H

        return (mu_t, Htilde_next), ll_from_alph(alph)

    mu0 = jnp.zeros((R, 1))
    Htilde0 = 0.0 * jnp.eye(R)
    (_, ll_seq) = jax.lax.scan(step, (mu0, Htilde0), jnp.arange(T))
    total_ll = jnp.sum(ll_seq)
    return -total_ll


def explicit_nll_inv_hessian(parameters, ctx: StartDetContext, scaling_code: ScalingCode):
    """NLL for inverse-Hessian filter."""
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

    def step(mu_prev, t):
        y_t   = Y[t].reshape(R, 1)
        xbeta = (X[t] @ beta).reshape(R, 1)
        Z1y   = Z1 @ y_t

        dev = Z1y - xbeta - mu_prev
        step_vec, alpha = _s_grad_invHess(dev, A, Z2, lam_inv, nu, delta)
        ll_t = ll_from_alph(alpha)

        mu_filt = mu_prev + kappa * step_vec
        mu_t = phi * mu_filt

        return mu_t, ll_t

    mu0 = jnp.zeros((R, 1))
    _, ll_seq = jax.lax.scan(step, mu0, jnp.arange(T))
    total_ll = jnp.sum(ll_seq)
    return -total_ll


# =============================================================================
# Evaluation (per start_mode)
# =============================================================================

def explicit_eval_standard(parameters, ctx: StartDetContext, scaling_code: ScalingCode, mu0):
    """Evaluation roll for standard scalings."""
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

    def step(mu_t, t):
        y_t   = Y[t].reshape(R, 1)
        xbeta = (X[t] @ beta).reshape(R, 1)
        yhat_t = jnp.linalg.solve(Z1, xbeta + mu_t).reshape(-1)
        dev   = (Z1 @ y_t) - xbeta - mu_t

        S, alph = _s_grad_explicit(dev, Z2, lam_inv, nu, scaling_code_int)
        ll_t = ll_from_alph(alph)

        mu_filt = mu_t + kappa * S
        mu_next = phi * mu_filt

        return mu_next, (mu_t.reshape(-1), ll_t, yhat_t, mu_filt)

    (mu_pred_last, (mu_seq_flat, ll_seq, yhat_seq, mu_filt_seq)) = lax.scan(
        step, mu0, jnp.arange(T)
    )
    mu_seq = mu_seq_flat.reshape(T, R)
    yhat = yhat_seq.reshape(T, R)
    mu_filt = mu_filt_seq.reshape(T, R)
    return mu_seq, ll_seq, yhat, mu_pred_last, mu_filt


def explicit_eval_ewma(parameters, ctx: StartDetContext, mu0, Htilde0):
    """Evaluation roll for EWMA-of-Hessian filter."""
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

    def step(carry, t):
        mu_prev, Htilde_t = carry

        y_t   = Y[t].reshape(R, 1)
        xbeta = (X[t] @ beta).reshape(R, 1)
        yhat_t = jnp.linalg.solve(Z1, xbeta + mu_prev).reshape(-1)

        dev  = (Z1 @ y_t) - xbeta - mu_prev
        g_t, alph, Av = _score_mu_and_alpha(dev, Z2, lam_inv, nu)

        Htilde_t, ew = spectral_floor_and_expm(Htilde_t, delta)
        step_vec = (kappa.reshape(-1) * (ew @ g_t)).reshape(R, 1)
        mu_filt = mu_prev + step_vec
        mu_t = phi * mu_filt

        cfac = (nu + R) / nu
        H    = (cfac / alph) * A - (2.0 * cfac / (nu * alph * alph)) * (Av @ Av.T)
        H    = 0.5 * (H + H.T)

        Htilde_next = gamma * Htilde_t + (1.0 - gamma) * H

        ll_t = ll_from_alph(alph)
        return (mu_t, Htilde_next), (mu_prev.reshape(-1), ll_t, yhat_t, mu_filt)

    (carry_last, (mu_seq_flat, ll_seq, yhat_seq, mu_filt_seq)) = jax.lax.scan(
        step, (mu0, Htilde0), jnp.arange(T)
    )
    mu_pred_last, Htilde_last = carry_last

    mu_seq = mu_seq_flat.reshape(T, R)
    yhat = yhat_seq.reshape(T, R)
    mu_filt = mu_filt_seq.reshape(T, R)
    return mu_seq, ll_seq, yhat, mu_pred_last, Htilde_last, mu_filt


def explicit_eval_local_hess(parameters, ctx: StartDetContext, mu0, Htilde0):
    """Evaluation roll for local-Hessian filter."""
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

    def step(carry, t):
        mu_prev, Htilde_t = carry

        y_t   = Y[t].reshape(R, 1)
        xbeta = (X[t] @ beta).reshape(R, 1)
        yhat_t = jnp.linalg.solve(Z1, xbeta + mu_prev).reshape(-1)

        dev  = (Z1 @ y_t) - xbeta - mu_prev
        g_t, alph, Av = _score_mu_and_alpha(dev, Z2, lam_inv, nu)

        Htilde_t, ew = spectral_floor_and_expm(Htilde_t, delta)
        step_vec = (kappa.reshape(-1) * (ew @ g_t)).reshape(R, 1)
        mu_filt = mu_prev + step_vec
        mu_t = phi * mu_filt

        cfac = (nu + R) / nu
        H    = (cfac / alph) * A - (2.0 * cfac / (nu * alph * alph)) * (Av @ Av.T)
        H    = 0.5 * (H + H.T)

        Htilde_next = kappa_h * jnp.eye(R, dtype=H.dtype) + gamma * Htilde_t + H

        ll_t = ll_from_alph(alph)
        return (mu_t, Htilde_next), (mu_prev.reshape(-1), ll_t, yhat_t, mu_filt)

    (carry_last, (mu_seq_flat, ll_seq, yhat_seq, mu_filt_seq)) = jax.lax.scan(
        step, (mu0, Htilde0), jnp.arange(T)
    )
    mu_pred_last, Htilde_last = carry_last

    mu_seq = mu_seq_flat.reshape(T, R)
    yhat = yhat_seq.reshape(T, R)
    mu_filt = mu_filt_seq.reshape(T, R)
    return mu_seq, ll_seq, yhat, mu_pred_last, Htilde_last, mu_filt


def explicit_eval_inv_hessian(parameters, ctx: StartDetContext, mu0):
    """Evaluation roll for inverse-Hessian filter."""
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

    def step(mu_prev, t):
        y_t   = Y[t].reshape(R, 1)
        xbeta = (X[t] @ beta).reshape(R, 1)
        Z1y   = Z1 @ y_t
        yhat_t = jnp.linalg.solve(Z1, xbeta + mu_prev).reshape(-1)

        dev = Z1y - xbeta - mu_prev
        step_vec, alpha = _s_grad_invHess(dev, A, Z2, lam_inv, nu, delta)
        ll_t = ll_from_alph(alpha)

        mu_filt = mu_prev + kappa * step_vec
        mu_t = phi * mu_filt

        return mu_t, (mu_prev.reshape(-1), ll_t, yhat_t, mu_filt)

    (mu_pred_last, (mu_seq_flat, ll_seq, yhat_seq, mu_filt_seq)) = jax.lax.scan(
        step, mu0, jnp.arange(T)
    )
    mu_seq = mu_seq_flat.reshape(T, R)
    yhat = yhat_seq.reshape(T, R)
    mu_filt = mu_filt_seq.reshape(T, R)
    return mu_seq, ll_seq, yhat, mu_pred_last, mu_filt


# =============================================================================
# Optimizer wrapper (unified)
# =============================================================================

def explicit_fit_jax(ctx: StartDetContext,
                     scaling_code: ScalingCode,
                     start_mode: str,
                     par0,
                     bounds,
                     maxiter: int = 1500):
    """Unified JAXopt L-BFGS-B wrapper for all start_mode variants."""
    if start_mode == "standard":
        fun = lambda th: explicit_nll_standard(th, ctx, scaling_code)
    elif start_mode == "ewma":
        fun = lambda th: explicit_nll_ewma(th, ctx, scaling_code)
    elif start_mode == "local_hess":
        fun = lambda th: explicit_nll_local_hess(th, ctx, scaling_code)
    elif start_mode == "inv_hessian":
        fun = lambda th: explicit_nll_inv_hessian(th, ctx, scaling_code)
    else:
        raise ValueError(f"Unknown start_mode '{start_mode}'")

    lb, ub = _bounds_to_box(bounds)
    solver = ScipyBoundedMinimize(
        method="l-bfgs-b",
        fun=fun,
        jit=True,
        maxiter=maxiter,
        options={'ftol': 1e-12, 'gtol': 1e-10},
    )
    x0 = jnp.asarray(par0, dtype=jnp.float64)
    params_opt, info = solver.run(x0, (lb, ub))
    return params_opt, info


# =============================================================================
# High-level helpers: fit + evaluate
# =============================================================================

def _resolve_scaling_code(scaling):
    """Convert user scaling input (str/int/ScalingCode) into ScalingCode."""
    if isinstance(scaling, str):
        try:
            return _SCALING_CODE_MAP[scaling]
        except KeyError:
            raise ValueError(
                f"Unknown scaling '{scaling}'. "
                f"Available: {list(_SCALING_CODE_MAP.keys())}"
            )
    elif isinstance(scaling, ScalingCode):
        return scaling
    else:
        return ScalingCode(int(scaling))


def fit_and_evaluate(Y, X, W,
                     scaling,
                     start_mode: str = "standard",
                     save_str: str = "start_params.pkl",
                     maxiter: int = 1500):
    """Fit explicit filter and compute in-sample performance metrics."""
    cT, cR = Y.shape
    ctx = StartDetContext(Y=Y, X=X, W1=W, W2=W)

    scaling_code = _resolve_scaling_code(scaling)

    # Start parameters from unified start_det module
    par0_np, bounds = build_start_params_startdet(
        Y, X, W, W,
        cb=False,
        scaling=scaling,
        start=None,
        save_str=save_str,
        start_mode=start_mode,
    )
    par0 = jnp.asarray(par0_np, dtype=jnp.float64)

    # Fit
    t0 = time.time()
    print('--- JAX L-BFGS-B fit (explicit NLL) ---')
    params_opt, state = explicit_fit_jax(ctx, scaling_code, start_mode, par0, bounds, maxiter=maxiter)
    print("Finished JAX fit. Final NLL:", float(state.fun_val))
    t1 = time.time()
    print(f'It took: {round(t1 - t0, 4)} seconds.')

    # In-sample eval
    mu0 = jnp.zeros((cR, 1))
    if start_mode in ("ewma", "local_hess"):
        Htilde0 = 0.0 * jnp.eye(cR)
        if start_mode == "ewma":
            mu_in, llik_in, yhat_in, _, _, mu_filt_in = explicit_eval_ewma(params_opt, ctx, mu0, Htilde0)
        else:  # local_hess
            mu_in, llik_in, yhat_in, _, _, mu_filt_in = explicit_eval_local_hess(params_opt, ctx, mu0, Htilde0)
    elif start_mode == "standard":
        mu_in, llik_in, yhat_in, _, mu_filt_in = explicit_eval_standard(params_opt, ctx, scaling_code, mu0)
    elif start_mode == "inv_hessian":
        mu_in, llik_in, yhat_in, _, mu_filt_in = explicit_eval_inv_hessian(params_opt, ctx, mu0)
    else:
        raise ValueError(f"Unknown start_mode '{start_mode}'")

    mu_in_np   = np.array(mu_in)
    llik_in_np = np.array(llik_in)
    yhat_in_np = np.array(yhat_in)

    avg_log_in = float(llik_in_np.mean())
    print('--- Evaluation ---')
    print(f'In-sample avg log-score:  {avg_log_in:.8f}')

    return {
        'state': state,
        'params_opt': np.array(params_opt),
        'final_nll': float(state.fun_val),
        'mu_in': mu_in_np,
        'llik_in': llik_in_np,
        'yhat_in': yhat_in_np,
        'avg_log_in': avg_log_in,
        'bounds': bounds,
        'par0': np.array(par0_np),
    }


def explicit_analyse_oos(Y_in, X_in, mu_in,
                         Y_out, X_out, mu_out,
                         W,
                         scaling,
                         start_mode: str,
                         cb_start: bool,
                         cb_general: bool,
                         save_str: str = "start_params.pkl",
                         maxiter: int = 1500):
    """In-/out-of-sample fit and evaluation, unified over start_mode."""
    cT, cR = Y_in.shape
    ctx_in = StartDetContext(Y=Y_in, X=X_in, W1=W, W2=W)
    ctx_out = StartDetContext(Y=Y_out, X=X_out, W1=W, W2=W)

    scaling_code = _resolve_scaling_code(scaling)

    # Start params
    par0_np, bounds = build_start_params_startdet(
        Y_in, X_in, W, W,
        cb=cb_start,
        scaling=scaling,
        start=None,
        save_str=save_str,
        start_mode=start_mode,
    )
    par0 = jnp.asarray(par0_np, dtype=jnp.float64)

    # Fit
    t_start = time.time()
    print('--- JAX L-BFGS-B fit (explicit NLL) ---')
    params_opt, state = explicit_fit_jax(ctx_in, scaling_code, start_mode, par0, bounds, maxiter=maxiter)
    t_end = time.time()
    print("Finished JAX fit. Final NLL:", float(state.fun_val))
    theta_opt = np.array(params_opt)
    print(f'It took: {round(t_end - t_start, 4)} seconds.')

    # In-sample eval
    mu0 = jnp.zeros((cR, 1))
    if start_mode in ("ewma", "local_hess"):
        Htilde0 = 0.0 * jnp.eye(cR)
        if start_mode == "ewma":
            mu_in_hat, llik_in, y_in_hat, mu_pred_last, Htilde_last, mu_filt_in_hat = explicit_eval_ewma(
                params_opt, ctx_in, mu0, Htilde0
            )
        else:
            mu_in_hat, llik_in, y_in_hat, mu_pred_last, Htilde_last, mu_filt_in_hat = explicit_eval_local_hess(
                params_opt, ctx_in, mu0, Htilde0
            )
    elif start_mode == "standard":
        mu_in_hat, llik_in, y_in_hat, mu_pred_last, mu_filt_in_hat = explicit_eval_standard(
            params_opt, ctx_in, scaling_code, mu0
        )
        Htilde_last = None
    elif start_mode == "inv_hessian":
        mu_in_hat, llik_in, y_in_hat, mu_pred_last, mu_filt_in_hat = explicit_eval_inv_hessian(
            params_opt, ctx_in, mu0
        )
        Htilde_last = None
    else:
        raise ValueError(f"Unknown start_mode '{start_mode}'")

    mu_in_hat_np = np.array(mu_in_hat)
    llik_in_np   = np.array(llik_in)
    y_in_hat_np  = np.array(y_in_hat)
    mu_filt_in_hat_np = np.array(mu_filt_in_hat)

    avg_log_in = float(llik_in_np.mean())

    err_y_in = np.asarray(Y_in) - y_in_hat_np
    mse_y_in = float(np.mean(err_y_in**2))
    mae_y_in = float(np.mean(np.abs(err_y_in)))

    if mu_in is not None:
        err_mu_in = np.asarray(mu_in) - mu_in_hat_np
        mse_mu_in = float(np.mean(err_mu_in**2))
        mae_mu_in = float(np.mean(np.abs(err_mu_in)))
    else:
        mse_mu_in = None
        mae_mu_in = None

    print('--- In-sample evaluation ---')
    print(f'In-sample avg log-score:  {avg_log_in:.8f}')

    # Out-of-sample eval
    if start_mode in ("ewma", "local_hess"):
        if start_mode == "ewma":
            mu_out_hat, llik_out, y_out_hat, _, _, mu_filt_out_hat = explicit_eval_ewma(
                params_opt, ctx_out, mu_pred_last, Htilde_last
            )
        else:
            mu_out_hat, llik_out, y_out_hat, _, _, mu_filt_out_hat = explicit_eval_local_hess(
                params_opt, ctx_out, mu_pred_last, Htilde_last
            )
    elif start_mode == "standard":
        mu_out_hat, llik_out, y_out_hat, _, mu_filt_out_hat = explicit_eval_standard(
            params_opt, ctx_out, scaling_code, mu_pred_last
        )
    elif start_mode == "inv_hessian":
        mu_out_hat, llik_out, y_out_hat, _, mu_filt_out_hat = explicit_eval_inv_hessian(
            params_opt, ctx_out, mu_pred_last
        )
    else:
        raise ValueError(f"Unknown start_mode '{start_mode}'")

    mu_out_hat_np = np.array(mu_out_hat)
    llik_out_np   = np.array(llik_out)
    y_out_hat_np  = np.array(y_out_hat)
    mu_filt_out_hat_np = np.array(mu_filt_out_hat)

    avg_log_out = float(llik_out_np.mean())

    err_y_out = np.asarray(Y_out) - y_out_hat_np
    mse_y_out = float(np.mean(err_y_out**2))
    mae_y_out = float(np.mean(np.abs(err_y_out)))

    if mu_out is not None:
        err_mu_out = np.asarray(mu_out) - mu_out_hat_np
        mse_mu_out = float(np.mean(err_mu_out**2))
        mae_mu_out = float(np.mean(np.abs(err_mu_out)))
    else:
        mse_mu_out = None
        mae_mu_out = None

    print('--- Out-of-sample evaluation ---')
    print(f'Out-sample avg log-score:  {avg_log_out:.8f}')

    return {
        'state': state,
        'opt': theta_opt,
        'params': np.array(params_opt),
        '-ll': float(state.fun_val),
        "mu_in_hat": mu_in_hat_np,
        "mu_out_hat": mu_out_hat_np,
        'mu_filt_in_hat': mu_filt_in_hat_np,
        'mu_filt_out_hat': mu_filt_out_hat_np,
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
        'avg_log_in': avg_log_in,
        'bounds': bounds,
        'par0': np.array(par0_np),
    }
