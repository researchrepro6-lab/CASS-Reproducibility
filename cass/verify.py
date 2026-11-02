"""Gradient verification: Taylor remainder test, component-wise finite differences
in scaled coordinates, RK4 step refinement, and an independent continuous
adjoint (DOP853 + AD Jacobians) with quadrature refinement."""
from __future__ import annotations

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np

# np.trapezoid exists only in NumPy >= 2.0; np.trapz is the NumPy 1.x name
_trapezoid = getattr(np, "trapezoid", None) or np.trapz
from scipy.integrate import solve_ivp

from . import model as Mdl
from .model import S_CAL


def taylor_test(w, gam, u0=Mdl.U0, cfg=Mdl.BASE, seed=0, hs=None):
    """r(h) = |J(u0 + h S d) - J(u0) - h g^T S d| for a random unit scaled direction d."""
    hs = np.logspace(-1, -6, 11) if hs is None else hs
    rng = np.random.default_rng(seed)
    d = rng.standard_normal(4)
    d /= np.linalg.norm(d)
    S = jnp.asarray(S_CAL)
    u0j, wj, gj = jnp.asarray(u0), jnp.asarray(w), jnp.asarray(gam)
    J0 = float(Mdl.objective(u0j, wj, gj, cfg))
    g = np.asarray(jax.grad(Mdl.objective)(u0j, wj, gj, cfg))
    slope = g @ (S_CAL * d)
    r0, r1 = [], []
    for h in hs:
        Jh = float(Mdl.objective(u0j + h * S * d, wj, gj, cfg))
        r0.append(abs(Jh - J0))
        r1.append(abs(Jh - J0 - h * slope))
    hs, r0, r1 = map(np.asarray, (hs, r0, r1))
    # fitted order on the asymptotic range (exclude round-off floor)
    mask = r1 > 1e-12
    order = np.polyfit(np.log(hs[mask]), np.log(r1[mask]), 1)[0]
    return dict(h=hs, r0=r0, r1=r1, order=float(order))


def fd_componentwise(w, gam, u0=Mdl.U0, cfg=Mdl.BASE, h=1e-4):
    """Central differences in scaled coordinates x (u = u0 + S x). Returns
    scaled adjoint gradient S g, FD estimate and component-wise relative errors
    |a_i - f_i| / |f_i|."""
    S = jnp.asarray(S_CAL)
    u0j, wj, gj = jnp.asarray(u0), jnp.asarray(w), jnp.asarray(gam)
    g = np.asarray(jax.grad(Mdl.objective)(u0j, wj, gj, cfg)) * S_CAL
    fd = np.zeros(4)
    for i in range(4):
        e = jnp.zeros(4).at[i].set(h)
        fd[i] = (float(Mdl.objective(u0j + S * e, wj, gj, cfg))
                 - float(Mdl.objective(u0j - S * e, wj, gj, cfg))) / (2 * h)
    rel = np.abs(g - fd) / np.abs(fd)
    return dict(adj=g, fd=fd, rel=rel, rel_norm=float(np.linalg.norm(g - fd) / np.linalg.norm(fd)))


def dt_refinement(w, gam, u0=Mdl.U0, steps=(500, 1000, 2000, 4000, 8000)):
    """Scaled gradient for successively finer RK4 grids; error vs the finest."""
    gs = []
    for n in steps:
        cfg = replace(Mdl.BASE, n_steps=n)
        gs.append(np.asarray(jax.grad(Mdl.objective)(jnp.asarray(u0), jnp.asarray(w),
                                                    jnp.asarray(gam), cfg)) * S_CAL)
    ref = gs[-1]
    err = [np.max(np.abs(g - ref) / np.abs(ref)) for g in gs[:-1]]
    return dict(steps=list(steps[:-1]), max_rel_err=err, grads=np.asarray(gs))


# ----------------------------------------------------------------------------
# Independent continuous adjoint (different solver, different discretisation)
# ----------------------------------------------------------------------------

def continuous_adjoint(w, gam, u0=Mdl.U0, n_quad=(501, 1001, 2001, 4001),
                       rtol=1e-12, atol=1e-14, cfg=Mdl.BASE):
    u0j, wj = jnp.asarray(u0), jnp.asarray(w)
    f = jax.jit(lambda y: Mdl.rhs(y, u0j, wj, cfg))
    A = jax.jit(jax.jacfwd(lambda y: Mdl.rhs(y, u0j, wj, cfg)))
    B = jax.jit(jax.jacfwd(lambda u, y: Mdl.rhs(y, u, wj, cfg)))

    fw = solve_ivp(lambda t, y: np.asarray(f(jnp.asarray(y))), (0, Mdl.T_F),
                   np.asarray(Mdl.y0()), method="DOP853", rtol=rtol, atol=atol,
                   dense_output=True)
    yT = fw.y[:, -1]
    muT = np.zeros(18)
    muT[0:3] = yT[0:3] - Mdl.P_TARGET
    bw = solve_ivp(lambda t, mu: -np.asarray(A(jnp.asarray(fw.sol(t)))).T @ mu,
                   (Mdl.T_F, 0), muT, method="DOP853", rtol=rtol, atol=atol,
                   dense_output=True)
    z = np.asarray(u0) / S_CAL
    greg = np.array([gam[0] * z[0], gam[1] * z[1], gam[1] * z[2], gam[1] * z[3]]) / S_CAL
    out = {}
    for n in n_quad:
        t = np.linspace(0, Mdl.T_F, n)
        integrand = np.stack([np.asarray(B(u0j, jnp.asarray(fw.sol(ti)))).T @ bw.sol(ti) for ti in t], 1)
        g = _trapezoid(integrand, t, axis=1) + greg
        out[n] = g * S_CAL
    return out
