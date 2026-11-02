"""Step 5 - robustness of the CASS structure to model choices.

Variants (LOW and HIGH cases):
  open-loop      baseline model of the paper
  rotor drag     + linear rotor drag  -k_rd (T/mg) R P R^T v_rel
  closed loop    attitude stabilisation + altitude hold; u is the feed-forward command
  near-optimum   baseline model, but CASS evaluated at the ensemble-mean optimum u*
                 (target reachable; nominal near an operating optimum)
"""
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cass  # noqa: E402,F401  (checks that the required packages are installed)
import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize

from cass import control as C, io, model as M, stats as ST

VARIANTS = {
    "open-loop": M.BASE,
    "rotor drag": replace(M.BASE, rotor_drag=True),
    "closed loop": replace(M.BASE, closed_loop=True),
}


def mean_optimum(W, gam, cfg):
    S = jnp.asarray(M.S_CAL)
    Wj, gj, u0 = jnp.asarray(W), jnp.asarray(gam), jnp.asarray(M.U0)
    f = jax.jit(jax.value_and_grad(lambda x: jnp.mean(M.J_all(u0 + S * x, Wj, gj, cfg))))
    res = minimize(lambda x: tuple(np.asarray(v, float) for v in f(jnp.asarray(x))),
                   np.zeros(4), jac=True, method="L-BFGS-B",
                   options=dict(maxiter=1000, ftol=1e-16, gtol=1e-10))
    return M.U0 + M.S_CAL * res.x, res


def diag(W, gam, u, cfg):
    Wj, gj, uj = jnp.asarray(W), jnp.asarray(gam), jnp.asarray(u)
    G = np.asarray(M.grad_all(uj, Wj, gj, cfg))
    J = np.asarray(M.J_all(uj, Wj, gj, cfg))
    s = ST.summary(G)
    wbar, Sw, _ = ST.wind_stats(W)
    K = np.asarray(M.K_matrix(uj, jnp.asarray(wbar), gj, cfg))
    e2, _ = ST.prop2_error(s["D"], K, Sw)
    bstar, _ = C.beta_star(s["c"], s["H"])
    return dict(J_mean=J.mean(), J_std=J.std(ddof=1), c_norm=np.linalg.norm(s["c"]),
                fro=s["fro"], lam1=s["lam1"], lam2=s["lam2"], top2_share=s["top2_share"],
                lam_hidden=s["lam_hidden"], kappa_hidden=s["kappa_hidden"],
                prop2_rel_err=e2, beta_star=bstar, v1=s["V"][:, 0])


def main():
    gam = io.gamma()
    cases = io.primary_cases()
    out = {}
    for vname, cfg in VARIANTS.items():
        out[vname] = {c: diag(d["W"], gam, M.U0, cfg) for c, d in cases.items()}
        print(vname, {c: (round(v["lam1"], 4), round(v["prop2_rel_err"], 4)) for c, v in out[vname].items()})
    out["near-optimum"] = {}
    for c, d in cases.items():
        ustar, res = mean_optimum(d["W"], gam, M.BASE)
        dd = diag(d["W"], gam, ustar, M.BASE)
        dd["u_star"] = ustar
        dd["converged"] = bool(res.success)
        out["near-optimum"][c] = dd
        print("near-optimum", c, ustar, dd["J_mean"], dd["c_norm"], dd["prop2_rel_err"])
    for v in out:
        out[v]["ratio_lam1"] = out[v]["HIGH"]["lam1"] / out[v]["LOW"]["lam1"]
        out[v]["angle_v1_LOW_vs_base"] = ST.angle(out[v]["LOW"]["v1"], out["open-loop"]["LOW"]["v1"])
        out[v]["angle_v1_HIGH_vs_base"] = ST.angle(out[v]["HIGH"]["v1"], out["open-loop"]["HIGH"]["v1"])
    io.dump(out, "variants.json")


if __name__ == "__main__":
    main()
