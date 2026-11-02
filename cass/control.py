"""Local control rules compared in the study.

All rules choose a scaled direction x in the box ||x||_inf <= 1 and apply
delta_u = alpha * S x (same feasible set for every rule).
"""
from __future__ import annotations

import cvxpy as cp
import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize

from . import model as Mdl
from .model import S_CAL


def _sqrt_psd(H, tol=0.0):
    lam, V = np.linalg.eigh((H + H.T) / 2)
    lam = np.clip(lam, tol, None)
    return V * np.sqrt(lam)            # L with H = L L^T


def psd_project(Q):
    lam, V = np.linalg.eigh((Q + Q.T) / 2)
    return (V * np.clip(lam, 0, None)) @ V.T


def beta_star(c, H):
    """Closed-form threshold below which the vertex v = -sign(c) solves
    min c^T x + beta sqrt(x^T H x) over the box (Proposition 3)."""
    v = -np.sign(c)
    Hv = H @ v
    sig = np.sqrt(max(v @ Hv, 0.0))
    ratios = [abs(c[i]) * sig / (v[i] * Hv[i]) for i in range(len(c)) if v[i] * Hv[i] > 0]
    return (min(ratios) if ratios else np.inf), v


def cass_socp(c, H, beta, alpha=1.0, Q=None):
    """min alpha c^T x + 0.5 alpha^2 x^T Q x + beta alpha sqrt(x^T H x), ||x||_inf <= 1.
    Q = None gives the first-order CASS rule (an SOCP). Solved with Clarabel."""
    d = len(c)
    x = cp.Variable(d)
    L = _sqrt_psd(H)
    obj = alpha * c @ x + beta * alpha * cp.norm(L.T @ x, 2)
    if Q is not None:
        LQ = _sqrt_psd(psd_project(Q))
        obj = obj + 0.5 * alpha ** 2 * cp.sum_squares(LQ.T @ x)
    prob = cp.Problem(cp.Minimize(obj), [cp.norm(x, "inf") <= 1])
    prob.solve(solver=cp.CLARABEL)
    return np.asarray(x.value).ravel(), prob


def cvar_rule(J0, Sg, alpha, Q, q=0.9):
    """CVaR_q of the member-wise second-order local model
    J_k0 + alpha s_k^T x + 0.5 alpha^2 x^T Q x  (Rockafellar-Uryasev form)."""
    n, d = Sg.shape
    x = cp.Variable(d)
    t = cp.Variable()
    LQ = _sqrt_psd(psd_project(Q))
    quad = 0.5 * alpha ** 2 * cp.sum_squares(LQ.T @ x)
    losses = J0 + alpha * Sg @ x
    obj = t + cp.sum(cp.pos(losses - t)) / ((1 - q) * n) + quad
    prob = cp.Problem(cp.Minimize(obj), [cp.norm(x, "inf") <= 1])
    prob.solve(solver=cp.CLARABEL)
    return np.asarray(x.value).ravel(), prob


def saa_rule(W, gam, alpha, u0=Mdl.U0, cfg=Mdl.BASE, x0=None):
    """Sample-average approximation: minimise the mean NONLINEAR objective of the
    training members over the box, with exact AD gradients (L-BFGS-B)."""
    S = jnp.asarray(S_CAL)
    Wj = jnp.asarray(W)
    gj = jnp.asarray(gam)
    u0j = jnp.asarray(u0)

    def f(x):
        u = u0j + alpha * S * x
        return jnp.mean(Mdl.J_all(u, Wj, gj, cfg))

    fg = jax.jit(jax.value_and_grad(f))

    def fun(x):
        v, g = fg(jnp.asarray(x))
        return float(v), np.asarray(g, float)

    x0 = np.zeros(4) if x0 is None else x0
    res = minimize(fun, x0, jac=True, method="L-BFGS-B", bounds=[(-1, 1)] * 4,
                   options=dict(maxiter=500, ftol=1e-15, gtol=1e-12))
    return res.x, res


def evaluate(x, W, gam, alpha, u0=Mdl.U0, cfg=Mdl.BASE, tau=None):
    """Nonlinear evaluation of delta_u = alpha S x on the members W."""
    u = jnp.asarray(u0) + alpha * jnp.asarray(S_CAL) * jnp.asarray(x)
    J = np.asarray(Mdl.J_all(u, jnp.asarray(W), jnp.asarray(gam), cfg))
    P = np.asarray(Mdl.terminal_positions(u, jnp.asarray(W), cfg))
    err = np.linalg.norm(P - Mdl.P_TARGET, axis=1)
    out = dict(mean=J.mean(), std=J.std(ddof=1), p90=np.percentile(J, 90),
               p95=np.percentile(J, 95), worst=J.max(), J=J, err=err)
    if tau is not None:
        out["exceed"] = float(np.mean(err > tau))
    return out
