"""
Quadrotor model, objective and exact derivatives for the CASS study.

Notation follows the manuscript:
  m      vehicle mass                     I_B   body inertia (diagonal)
  S_ref  drag reference area              k     ensemble member index
  gamma  regularization weights           mu(t) adjoint state (continuous check only)
  S      characteristic control scaling   x     scaled control, u = S x

A single solver is used everywhere: classical fixed-step RK4 with N steps on
[0, T_f].  Gradients are obtained by reverse-mode automatic differentiation
through the RK4 recursion, i.e. the exact discrete adjoint of the discretized
problem.  Hessians and the mixed derivative K = d g / d w use forward-over-
reverse AD.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
M = 1.5                     # kg
G = 9.81                    # m s^-2
I_B = np.array([0.03, 0.03, 0.05])   # kg m^2
RHO = 1.225                 # kg m^-3
C_D = 1.2
S_REF = 0.05                # m^2
T_F = 2.0                   # s
P_TARGET = np.array([2.0, 1.0, 0.0])
ALPHA_REF = 0.5             # rad s^-2, reference angular acceleration
# Characteristic control scales: hover thrust and the torque that produces
# ALPHA_REF about each body axis  ->  (14.715, 0.015, 0.015, 0.025)
S_CAL = np.array([M * G, I_B[0] * ALPHA_REF, I_B[1] * ALPHA_REF, I_B[2] * ALPHA_REF])
U0 = np.array([M * G, 0.001, 0.0005, 0.0002])   # hover-like nominal control
N_STEPS = 2000              # RK4 steps (dt = 1e-3 s)
RHO_BAL = 0.10              # gradient-balancing ratio for the regularization


@dataclass(frozen=True)
class Config:
    """Model variant switches (hashable -> usable as a static JAX argument)."""
    rotor_drag: bool = False
    k_rd: float = 0.35          # N s m^-1, linear rotor-drag coefficient (variant)
    closed_loop: bool = False
    k_R: float = 0.5            # N m rad^-1 attitude stiffness (variant)
    k_w: float = 0.17           # N m s rad^-1 rate damping (variant)
    k_pz: float = 4.0           # s^-2 altitude stiffness (variant)
    k_dz: float = 4.0           # s^-1 altitude damping (variant)
    n_steps: int = N_STEPS


BASE = Config()

# ---------------------------------------------------------------------------
# Dynamics
# ---------------------------------------------------------------------------

def _skew(w):
    return jnp.array([[0.0, -w[2], w[1]],
                      [w[2], 0.0, -w[0]],
                      [-w[1], w[0], 0.0]])


def rhs(y, u, w2, cfg: Config = BASE):
    """Right-hand side f(y, u; w) with y = (p, v, vec(R), omega) in R^18."""
    p = y[0:3]
    v = y[3:6]
    R = y[6:15].reshape(3, 3)
    om = y[15:18]
    w = jnp.array([w2[0], w2[1], 0.0])
    I = jnp.asarray(I_B)

    T = u[0]
    tau = u[1:4]
    if cfg.closed_loop:
        # attitude stabilisation towards R_d = I and altitude hold; u is the
        # feed-forward command chosen by the planner
        e_R = 0.5 * jnp.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
        tau = tau - cfg.k_R * e_R - cfg.k_w * om + jnp.cross(om, I * om)
        T = T + M * (-cfg.k_pz * p[2] - cfg.k_dz * v[2])

    v_rel = v - w
    speed = jnp.sqrt(v_rel @ v_rel + 1e-12)
    F = -0.5 * RHO * C_D * S_REF * speed * v_rel
    if cfg.rotor_drag:
        P = jnp.diag(jnp.array([1.0, 1.0, 0.0]))
        F = F - cfg.k_rd * (T / (M * G)) * (R @ P @ R.T @ v_rel)

    e3 = jnp.array([0.0, 0.0, 1.0])
    dp = v
    dv = (T / M) * R[:, 2] - G * e3 + F / M
    dR = R @ _skew(om)
    dom = (tau - jnp.cross(om, I * om)) / I
    return jnp.concatenate([dp, dv, dR.reshape(-1), dom])


def y0():
    return jnp.concatenate([jnp.zeros(6), jnp.eye(3).reshape(-1), jnp.zeros(3)])


@partial(jax.jit, static_argnames=("cfg",))
def final_state(u, w2, cfg: Config = BASE):
    dt = T_F / cfg.n_steps

    def step(y, _):
        k1 = rhs(y, u, w2, cfg)
        k2 = rhs(y + 0.5 * dt * k1, u, w2, cfg)
        k3 = rhs(y + 0.5 * dt * k2, u, w2, cfg)
        k4 = rhs(y + dt * k3, u, w2, cfg)
        return y + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4), None

    yT, _ = jax.lax.scan(step, y0(), None, length=cfg.n_steps)
    return yT


@partial(jax.jit, static_argnames=("cfg",))
def trajectory(u, w2, cfg: Config = BASE):
    dt = T_F / cfg.n_steps

    def step(y, _):
        k1 = rhs(y, u, w2, cfg)
        k2 = rhs(y + 0.5 * dt * k1, u, w2, cfg)
        k3 = rhs(y + 0.5 * dt * k2, u, w2, cfg)
        k4 = rhs(y + dt * k3, u, w2, cfg)
        yn = y + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        return yn, yn

    _, ys = jax.lax.scan(step, y0(), None, length=cfg.n_steps)
    return jnp.vstack([y0()[None, :], ys])


# ---------------------------------------------------------------------------
# Objective
# ---------------------------------------------------------------------------

def tracking(u, w2, cfg: Config = BASE):
    p = final_state(u, w2, cfg)[0:3]
    d = p - jnp.asarray(P_TARGET)
    return 0.5 * d @ d


def regularization(u, gam):
    z = u / jnp.asarray(S_CAL)
    return 0.5 * gam[0] * z[0] ** 2 + 0.5 * gam[1] * (z[1:] @ z[1:])


def objective(u, w2, gam, cfg: Config = BASE):
    return tracking(u, w2, cfg) + regularization(u, gam)


@partial(jax.jit, static_argnames=("cfg",))
def J_all(u, W, gam, cfg: Config = BASE):
    return jax.vmap(lambda w: objective(u, w, gam, cfg))(W)


@partial(jax.jit, static_argnames=("cfg",))
def grad_all(u, W, gam, cfg: Config = BASE):
    """Member-wise gradients g_k = dJ_k/du (exact discrete adjoint), shape (E, 4)."""
    return jax.vmap(lambda w: jax.grad(objective)(u, w, gam, cfg))(W)


@partial(jax.jit, static_argnames=("cfg",))
def track_grad_all(u, W, cfg: Config = BASE):
    return jax.vmap(lambda w: jax.grad(tracking)(u, w, cfg))(W)


@partial(jax.jit, static_argnames=("cfg",))
def hess_all(u, W, gam, cfg: Config = BASE):
    return jax.vmap(lambda w: jax.hessian(objective)(u, w, gam, cfg))(W)


@partial(jax.jit, static_argnames=("cfg",))
def K_matrix(u, wbar, gam, cfg: Config = BASE):
    """Mixed derivative K = d g / d w (4 x 2) at the ensemble-mean wind."""
    return jax.jacfwd(lambda w: jax.grad(objective)(u, w, gam, cfg))(wbar)


@partial(jax.jit, static_argnames=("cfg",))
def terminal_positions(u, W, cfg: Config = BASE):
    return jax.vmap(lambda w: final_state(u, w, cfg)[0:3])(W)


# ---------------------------------------------------------------------------
# Regularization weights by gradient balancing
# ---------------------------------------------------------------------------

def balance_gamma(W_ref, u=U0, ratio=RHO_BAL, cfg: Config = BASE):
    """gamma such that, at u, the regularization gradient equals `ratio` times
    the ensemble-mean tracking gradient, separately for the thrust block and
    the torque block (Euclidean norms in physical units)."""
    gt = np.asarray(track_grad_all(jnp.asarray(u), jnp.asarray(W_ref), cfg)).mean(0)
    z = np.asarray(u) / S_CAL
    # d/du of 0.5*gam*z^2 = gam * z / S
    unit_T = abs(z[0] / S_CAL[0])
    unit_tau = np.linalg.norm(z[1:] / S_CAL[1:])
    gam_T = ratio * abs(gt[0]) / unit_T
    gam_tau = ratio * np.linalg.norm(gt[1:]) / unit_tau
    return np.array([gam_T, gam_tau]), gt
