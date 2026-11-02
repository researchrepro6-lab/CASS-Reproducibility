"""Step 3 - CASS statistics for every timestamp (62) at 32 locations:
the study site, its 5 x 5 full-resolution neighbourhood and 7 contrasting regimes.
Outputs results/multicase.npz and results/multicase.json (correlations)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cass  # noqa: E402,F401  (checks that the required packages are installed)
import jax.numpy as jnp
import numpy as np
from scipy.stats import spearmanr

from cass import io, model as M, stats as ST


def case_stats(W, gam):
    Wj, gj, u0 = jnp.asarray(W), jnp.asarray(gam), jnp.asarray(M.U0)
    G = np.asarray(M.grad_all(u0, Wj, gj))
    J = np.asarray(M.J_all(u0, Wj, gj))
    s = ST.summary(G)
    wbar, Sw, spread = ST.wind_stats(W)
    K = np.asarray(M.K_matrix(u0, jnp.asarray(wbar), gj))
    e2, _ = ST.prop2_error(s["D"], K, Sw)
    e_lin, r2_lin = ST.feature_fit_error(G, W)
    e_drag, r2_drag = ST.feature_fit_error(G, ST.drag_feature(W))
    return [spread, np.linalg.det(Sw), np.trace(Sw), s["fro"], s["lam1"], s["lam2"],
            s["lam_hidden"], s["rho_hidden"], s["kappa_hidden"], s["top2_share"], e2,
            J.std(ddof=1), np.linalg.norm(wbar), e_lin, e_drag, float(np.mean(r2_drag)),
            spread / np.linalg.norm(wbar)]


COLS = ["spread", "detSw", "trSw", "fro", "lam1", "lam2", "lam_hidden", "rho_hidden",
        "kappa_hidden", "top2_share", "prop2_err", "J_std", "wbar_speed", "eps_linfit",
        "eps_drag", "loo_r2_drag", "spread_ratio"]


def main():
    import os
    gam = io.gamma()
    z = io.load_winds()
    T = 2 if os.environ.get("CASS_QUICK") else len(z["times"])
    # site neighbourhood (includes the site at [2,2])
    nb = z["nbhd"]
    rows = []
    for t in range(T):
        for i in range(5):
            for j in range(5):
                rows.append([t, 0, i, j] + case_stats(nb[t, i, j], gam))
        print("nbhd t", t, flush=True)
    for L in range(z["locs"].shape[0]):
        for t in range(T):
            rows.append([t, 1 + L, -1, -1] + case_stats(z["locs"][L, t], gam))
        print("loc", L, flush=True)
    A = np.asarray(rows, float)
    np.savez_compressed(io.RES / "multicase.npz", A=A, cols=np.array(["t", "group", "i", "j"] + COLS))

    def corr(mask, x, y):
        r = spearmanr(A[mask, 4 + COLS.index(x)], A[mask, 4 + COLS.index(y)])
        return float(r.statistic)

    site = (A[:, 1] == 0) & (A[:, 2] == 2) & (A[:, 3] == 2)
    allm = np.ones(len(A), bool)
    res = {}
    for name, m in (("site", site), ("all", allm)):
        res[name] = dict(
            n=int(m.sum()),
            spearman_lam1_detSw=corr(m, "lam1", "detSw"),
            spearman_lam1_trSw=corr(m, "lam1", "trSw"),
            spearman_lam1_spread=corr(m, "lam1", "spread"),
            spearman_fro_spread=corr(m, "fro", "spread"),
            spearman_hidden_spread=corr(m, "lam_hidden", "spread"),
            spearman_Jstd_lam1=corr(m, "J_std", "lam1"),
            prop2_err_median=float(np.median(A[m, 4 + COLS.index("prop2_err")])),
            prop2_err_p95=float(np.percentile(A[m, 4 + COLS.index("prop2_err")], 95)),
            prop2_err_max=float(np.max(A[m, 4 + COLS.index("prop2_err")])),
            top2_share_min=float(np.min(A[m, 4 + COLS.index("top2_share")])),
            rho_hidden_median=float(np.median(A[m, 4 + COLS.index("rho_hidden")])),
            rho_hidden_range=[float(np.min(A[m, 4 + COLS.index("rho_hidden")])),
                              float(np.max(A[m, 4 + COLS.index("rho_hidden")]))],
            kappa_hidden_median=float(np.median(A[m, 4 + COLS.index("kappa_hidden")])),
            kappa_hidden_p95=float(np.percentile(A[m, 4 + COLS.index("kappa_hidden")], 95)),
            eps_linfit_median=float(np.median(A[m, 4 + COLS.index("eps_linfit")])),
            eps_drag_median=float(np.median(A[m, 4 + COLS.index("eps_drag")])),
            eps_drag_p95=float(np.percentile(A[m, 4 + COLS.index("eps_drag")], 95)),
            loo_r2_drag_median=float(np.median(A[m, 4 + COLS.index("loo_r2_drag")])),
            spearman_prop2_ratio=corr(m, "prop2_err", "spread_ratio"),
            frac_ratio_gt_half=float(np.mean(A[m, 4 + COLS.index("spread_ratio")] > 0.5)),
            prop2_err_median_ratio_lt_02=float(np.median(A[m & (A[:, 4 + COLS.index("spread_ratio")] < 0.2), 4 + COLS.index("prop2_err")])) if np.any(m & (A[:, 4 + COLS.index("spread_ratio")] < 0.2)) else float("nan"),
            n_ratio_lt_02=int(np.sum(m & (A[:, 4 + COLS.index("spread_ratio")] < 0.2))),
        )
    # log-log fit lam1 ~ trSw (power law)
    x = np.log(A[:, 4 + COLS.index("trSw")])
    y = np.log(A[:, 4 + COLS.index("lam1")])
    res["loglog_lam1_trSw_slope"] = float(np.polyfit(x, y, 1)[0])
    io.dump(res, "multicase.json")
    print(res)


if __name__ == "__main__":
    main()
