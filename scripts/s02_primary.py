"""Step 2 - CASS diagnostics for the two primary cases (LOW, HIGH).

Outputs results/primary.json and data/processed/gradients_{LOW,HIGH}.npz
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cass  # noqa: E402,F401  (checks that the required packages are installed)
import jax.numpy as jnp
import numpy as np

from cass import control as C, io, model as M, stats as ST

ALPHA = 0.01
ALPHAS = ALPHA * 2.0 ** -np.arange(0, 9)


def alpha_sweep(W, gam, x, c, H):
    u0 = jnp.asarray(M.U0)
    J0 = np.asarray(M.J_all(u0, jnp.asarray(W), jnp.asarray(gam)))
    rows = []
    for a in ALPHAS:
        u = u0 + a * jnp.asarray(M.S_CAL) * jnp.asarray(x)
        dJ = np.asarray(M.J_all(u, jnp.asarray(W), jnp.asarray(gam))) - J0
        pm, am = a * c @ x, dJ.mean()
        ps, as_ = a * np.sqrt(max(x @ H @ x, 0)), dJ.std(ddof=1)
        rows.append(dict(alpha=a, pred_mean=pm, act_mean=am, err_mean=abs(am - pm),
                         pred_std=ps, act_std=as_, err_std=abs(as_ - ps)))
    al = np.array([r["alpha"] for r in rows])
    fit = lambda e: float(np.polyfit(np.log(al), np.log(np.maximum(e, 1e-300)), 1)[0])
    return dict(rows=rows,
                slope_mean=fit(np.array([r["err_mean"] for r in rows])),
                slope_std=fit(np.array([r["err_std"] for r in rows])))


def main():
    gam = io.gamma()
    cases = io.primary_cases()
    out = {}
    for case, d in cases.items():
        W = d["W"]
        Wj, gj, u0 = jnp.asarray(W), jnp.asarray(gam), jnp.asarray(M.U0)
        G = np.asarray(M.grad_all(u0, Wj, gj))
        Gt = np.asarray(M.track_grad_all(u0, Wj))
        J0 = np.asarray(M.J_all(u0, Wj, gj))
        P = np.asarray(M.terminal_positions(u0, Wj))
        np.savez_compressed(io.ROOT / f"data/processed/gradients_{case}.npz", W=W, G=G,
                            G_tracking=Gt, J0=J0, P_terminal=P, gamma=gam, u0=M.U0, S=M.S_CAL,
                            time=d["time"])

        s = ST.summary(G)
        s0 = ST.summary(G, ddof=0)
        wbar, Sw, spread = ST.wind_stats(W)
        K = np.asarray(M.K_matrix(u0, jnp.asarray(wbar), gj))
        e2, Hlin = ST.prop2_error(s["D"], K, Sw)
        eps_linfit, r2_linfit = ST.feature_fit_error(G, W)
        eps_drag, r2_drag = ST.feature_fit_error(G, ST.drag_feature(W))
        # drag-feature representation of H for plotting
        Gs = G * M.S_CAL
        Pd = ST.drag_feature(W) - ST.drag_feature(W).mean(0)
        Bd, *_ = np.linalg.lstsq(Pd, Gs - Gs.mean(0), rcond=None)
        H_drag = (Pd @ Bd).T @ (Pd @ Bd) / (len(W) - 1)
        bstar, vbox = C.beta_star(s["c"], s["H"])
        boot = ST.bootstrap(G, B=2000, seed=1)
        loo = ST.leave_one_out(G)
        subs = ST.random_subsets(G, n_draw=1000, seed=2)
        scl = ST.scaling_sensitivity(G, n_draw=500, seed=3)

        # principal directions in physical coordinates (normalised)
        v1_phys = M.S_CAL * s["V"][:, 0]
        v1_phys /= np.linalg.norm(v1_phys)
        # eigen-directions of the linearised operator vs. wind principal axes
        lw, Vw = np.linalg.eigh(Sw)
        kw = np.diag(M.S_CAL) @ K @ Vw[:, -1]
        ang_K = ST.angle(kw, s["V"][:, 0])

        x_box = -np.sign(s["c"])
        x_mean = -s["c"] / np.max(np.abs(s["c"]))
        sweep = {"box": alpha_sweep(W, gam, x_box, s["c"], s["H"]),
                 "mean_dir": alpha_sweep(W, gam, x_mean, s["c"], s["H"]),
                 "hidden": alpha_sweep(W, gam, s["x_hidden"] / np.max(np.abs(s["x_hidden"])), s["c"], s["H"])}

        r_hidden = G @ (M.S_CAL * s["x_hidden"])
        out[case] = dict(
            time=d["time"], spread=spread, wbar=wbar, Sigma_w=Sw, det_Sigma_w=np.linalg.det(Sw),
            J_mean=J0.mean(), J_std=J0.std(ddof=1), J_min=J0.min(), J_max=J0.max(),
            pos_std=P.std(0, ddof=1), pos_disp=np.linalg.norm(P.std(0, ddof=1)),
            err_mean=np.linalg.norm(P - M.P_TARGET, axis=1).mean(),
            gbar=s["gbar"], gbar_tracking=Gt.mean(0), c=s["c"], H=s["H"], eig=s["eig"],
            V=s["V"], v1_phys=v1_phys, fro=s["fro"], fro_ddof0=s0["fro"],
            raw_fro=np.linalg.norm(s["D"], "fro"),
            lam1=s["lam1"], lam2=s["lam2"], lam_hidden=s["lam_hidden"],
            rho_hidden=s["rho_hidden"], kappa_hidden=s["kappa_hidden"],
            sigma_mean_dir=s["sigma_mean_dir"], top2_share=s["top2_share"],
            c_norm=np.linalg.norm(s["c"]), x_hidden=s["x_hidden"],
            hidden_resp=dict(mean=r_hidden.mean(), std=r_hidden.std(ddof=1),
                             min=r_hidden.min(), max=r_hidden.max()),
            prop1_ok=bool(s["lam2"] - 1e-12 <= s["lam_hidden"] <= s["lam1"] + 1e-12),
            K=K, prop2_rel_err=e2, H_lin=Hlin, angle_v1_vs_K_wind_axis=ang_K,
            eps_linfit=eps_linfit, r2_linfit=r2_linfit, eps_drag=eps_drag, r2_drag=r2_drag,
            H_drag=H_drag, wbar_speed=np.linalg.norm(wbar),
            spread_ratio=spread / np.linalg.norm(wbar),
            beta_star=bstar, x_box=x_box,
            bootstrap=boot, loo=dict(values=loo, max_abs=float(np.max(np.abs(loo))),
                                     argmax=int(np.argmax(np.abs(loo))) + 1,
                                     top3=(np.argsort(-np.abs(loo))[:3] + 1).tolist()),
            subsets=subs, scaling=scl, alpha_sweep=sweep)
        print(case, "fro %.4g lam1 %.4g lam2 %.4g hidden %.4g prop2 %.3f beta* %.4g" %
              (s["fro"], s["lam1"], s["lam2"], s["lam_hidden"], e2, bstar))
    out["ratios"] = dict(
        fro=out["HIGH"]["fro"] / out["LOW"]["fro"],
        lam1=out["HIGH"]["lam1"] / out["LOW"]["lam1"],
        hidden_rms=np.sqrt(out["HIGH"]["lam_hidden"] / out["LOW"]["lam_hidden"]),
        spread=out["HIGH"]["spread"] / out["LOW"]["spread"],
        detSw=out["HIGH"]["det_Sigma_w"] / out["LOW"]["det_Sigma_w"],
        J_std=out["HIGH"]["J_std"] / out["LOW"]["J_std"])
    io.dump(out, "primary.json")


if __name__ == "__main__":
    main()
