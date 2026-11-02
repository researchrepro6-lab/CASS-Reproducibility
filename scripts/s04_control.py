"""Step 4 - fair, out-of-sample comparison of local control rules.

For each split, 25 members are used to build the rule (training) and the other
25 members are used only for nonlinear evaluation (testing). All rules share
the same feasible set ||x||_inf <= 1 with delta_u = alpha S x.

Rules
  nominal        x = 0
  mean-box       beta = 0 solution of the box problem, x = -sign(c)
  mean-dir       steepest mean-descent direction scaled to the box, -c/||c||_inf
  CASS-1 (b)     first-order CASS SOCP, beta in {1.645, 2 beta*, 5 beta*}
  mean-2         second-order mean model (ensemble-mean Hessian), beta = 0
  CASS-2         second-order mean model + beta sqrt(x^T H x), beta = 1.645
  SAA            minimise the mean nonlinear training objective (L-BFGS-B, exact gradients)
  CVaR-0.9       CVaR of member-wise second-order local models (Rockafellar-Uryasev)
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cass  # noqa: E402,F401  (checks that the required packages are installed)
import jax.numpy as jnp
import numpy as np

from cass import control as C, io, model as M, stats as ST

import os
ALPHA = 0.01
QUICK = bool(os.environ.get("CASS_QUICK"))
BETA_Q = 1.645


def run_split(W, G, J0, Hs, tr, te, gam):
    Gt, Wt = G[tr], W[tr]
    s = ST.summary(Gt)
    c, H = s["c"], s["H"]
    Sg = Gt * M.S_CAL
    Q = np.diag(M.S_CAL) @ Hs[tr].mean(0) @ np.diag(M.S_CAL)
    bstar, _ = C.beta_star(c, H)
    rules = {"nominal": np.zeros(4), "mean-box": -np.sign(c), "mean-dir": -c / np.max(np.abs(c))}
    rules["CASS-1 (b=1.645)"] = C.cass_socp(c, H, BETA_Q)[0]
    if np.isfinite(bstar):
        rules["CASS-1 (b=2b*)"] = C.cass_socp(c, H, 2 * bstar)[0]
        rules["CASS-1 (b=5b*)"] = C.cass_socp(c, H, 5 * bstar)[0]
    rules["mean-2"] = C.cass_socp(c, H, 0.0, alpha=ALPHA, Q=Q)[0]
    rules["CASS-2 (b=1.645)"] = C.cass_socp(c, H, BETA_Q, alpha=ALPHA, Q=Q)[0]
    rules["SAA"] = C.saa_rule(Wt, gam, ALPHA)[0]
    rules["CVaR-0.9"] = C.cvar_rule(J0[tr], Sg, ALPHA, Q)[0]

    base_tr = C.evaluate(np.zeros(4), Wt, gam, ALPHA)
    tau = np.percentile(base_tr["err"], 90)
    res = {}
    for name, x in rules.items():
        e = C.evaluate(x, W[te], gam, ALPHA, tau=tau)
        res[name] = dict(x=x, xnorm2=float(np.linalg.norm(x)), mean=e["mean"], std=e["std"],
                         p90=e["p90"], p95=e["p95"], worst=e["worst"], exceed=e["exceed"])
    return res, bstar


def summarize(splits):
    names = list(splits[0].keys())
    out = {}
    for n in names:
        rows = [s[n] for s in splits if n in s]
        base = [s["nominal"] for s in splits if n in s]
        d = lambda k: np.array([r[k] - b[k] for r, b in zip(rows, base)])
        out[n] = dict(
            n=len(rows),
            d_mean=float(d("mean").mean()), d_mean_se=float(d("mean").std(ddof=1) / np.sqrt(len(rows))),
            d_std=float(d("std").mean()), d_p95=float(d("p95").mean()), d_worst=float(d("worst").mean()),
            exceed=float(np.mean([r["exceed"] for r in rows])),
            xnorm2=float(np.mean([r["xnorm2"] for r in rows])),
            x_mean=np.mean([r["x"] for r in rows], 0),
            win_mean=float(np.mean(d("mean") < 0)), win_p95=float(np.mean(d("p95") < 0)))
    return out


def run_case(W, gam, n_splits, seed):
    Wj, gj, u0 = jnp.asarray(W), jnp.asarray(gam), jnp.asarray(M.U0)
    G = np.asarray(M.grad_all(u0, Wj, gj))
    J0 = np.asarray(M.J_all(u0, Wj, gj))
    Hs = np.asarray(M.hess_all(u0, Wj, gj))
    rng = np.random.default_rng(seed)
    E = len(W)
    splits, bstars = [], []
    for _ in range(n_splits):
        perm = rng.permutation(E)
        r, b = run_split(W, G, J0, Hs, perm[:E // 2], perm[E // 2:], gam)
        splits.append(r)
        bstars.append(b)
    return summarize(splits), np.asarray(bstars), Hs.mean(0)


def main():
    gam = io.gamma()
    out = {}
    for case, d in io.primary_cases().items():
        t0 = time.time()
        summ, bst, Hbar = run_case(d["W"], gam, n_splits=4 if QUICK else 100, seed=10)
        out[case] = dict(summary=summ, beta_star_median=float(np.median(bst)),
                         mean_hessian=Hbar, scaled_mean_hessian_diag=np.diag(Hbar) * M.S_CAL ** 2)
        print(case, "%.1fs" % (time.time() - t0), flush=True)
        for n, v in summ.items():
            print("  %-18s dMean %+.4f dStd %+.4f dP95 %+.4f exceed %.2f |x| %.2f" %
                  (n, v["d_mean"], v["d_std"], v["d_p95"], v["exceed"], v["xnorm2"]))
    io.dump(out, "control_primary.json")

    # all 62 site timestamps, 10 splits each
    z = io.load_winds()
    allres = []
    for t in range(3 if QUICK else len(z["times"])):
        summ, bst, _ = run_case(np.asarray(z["site"][t], float), gam, n_splits=2 if QUICK else 10, seed=100 + t)
        allres.append(dict(t=t, summary=summ, beta_star_median=float(np.median(bst))))
        print("t", t, flush=True)
    names = list(allres[0]["summary"].keys())
    agg = {}
    for n in names:
        v = [r["summary"][n] for r in allres if n in r["summary"]]
        agg[n] = dict(n_cases=len(v),
                      d_mean=float(np.mean([x["d_mean"] for x in v])),
                      d_std=float(np.mean([x["d_std"] for x in v])),
                      d_p95=float(np.mean([x["d_p95"] for x in v])),
                      d_worst=float(np.mean([x["d_worst"] for x in v])),
                      exceed=float(np.mean([x["exceed"] for x in v])),
                      frac_cases_improve_mean=float(np.mean([x["d_mean"] < 0 for x in v])),
                      frac_cases_improve_p95=float(np.mean([x["d_p95"] < 0 for x in v])))
    io.dump(dict(per_time=allres, aggregate=agg), "control_alltimes.json")


if __name__ == "__main__":
    main()
