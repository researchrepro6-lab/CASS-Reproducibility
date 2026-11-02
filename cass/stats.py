"""CASS statistics: dispersion matrices, hidden dispersion, Proposition 1/2 checks,
bootstrap confidence intervals, influence diagnostics and random-subset analysis."""
from __future__ import annotations

import numpy as np
from scipy.linalg import eigh

from .model import S_CAL


def cass(G, ddof=1):
    """Return gbar and D = (1/(E-ddof)) sum (g_k - gbar)(g_k - gbar)^T."""
    G = np.asarray(G, float)
    gbar = G.mean(0)
    C = G - gbar
    D = C.T @ C / (G.shape[0] - ddof)
    return gbar, D


def scaled(gbar, D, S=S_CAL):
    """Scaled quantities c = S^T gbar, H = S^T D S (characteristic control coordinates;
    S is diagonal, so S^T = S)."""
    S = np.diag(S)
    return S.T @ gbar, S.T @ D @ S


def _is_zero(c, H):
    """Mean gradient numerically zero (round-off level relative to the dispersion)."""
    return np.linalg.norm(c) <= 1e-14 * max(1.0, np.sqrt(np.trace(H)))


def hidden(H, c):
    """lambda_hidden = max x^T H x s.t. c^T x = 0, ||x|| = 1, and its maximiser.
    If c = 0 the constraint is void and lambda_hidden = lambda_1 (Remark 1 of the paper)."""
    if _is_zero(c, H):
        lam, V = eigh(H)
        return lam[-1], V[:, -1]
    n = c / np.linalg.norm(c)
    # orthonormal basis of c-perp
    Q, _ = np.linalg.qr(np.column_stack([n, np.eye(len(c))]))
    B = Q[:, 1:len(c)]
    lam, V = eigh(B.T @ H @ B)
    return lam[-1], B @ V[:, -1]


def summary(G, S=S_CAL, ddof=1):
    gbar, D = cass(G, ddof)
    c, H = scaled(gbar, D, S)
    lam, V = eigh(H)
    lam = lam[::-1]
    V = V[:, ::-1]
    lh, xh = hidden(H, c)
    zero = _is_zero(c, H)
    cn = np.linalg.norm(c)
    u_c = np.zeros_like(c) if zero else c / cn
    return dict(
        gbar=gbar, D=D, c=c, H=H, eig=lam, V=V,
        fro=np.linalg.norm(H, "fro"), lam1=lam[0], lam2=lam[1],
        lam_hidden=lh, x_hidden=xh,
        rho_hidden=lh / lam[0],                       # where in [lam2, lam1] relative to lam1
        kappa_hidden=np.inf if zero else np.sqrt(lh) / cn,  # hidden RMS slope / mean slope;
        # undefined (reported as inf) when the mean gradient vanishes
        sigma_mean_dir=np.nan if zero else np.sqrt(u_c @ H @ u_c),
        top2_share=(lam[0] + lam[1]) / lam.sum(),
    )


def wind_stats(W, ddof=1):
    W = np.asarray(W, float)
    wbar = W.mean(0)
    Sw = np.cov(W.T, ddof=ddof)
    spread = np.mean(np.linalg.norm(W - wbar, axis=1))
    return wbar, Sw, spread


def prop2_error(D, K, Sw, S=S_CAL):
    """Relative Frobenius error of the linearisation D ~ K Sw K^T in scaled coordinates."""
    Sd = np.diag(S)
    H = Sd @ D @ Sd
    Hl = Sd @ (K @ Sw @ K.T) @ Sd
    return np.linalg.norm(H - Hl, "fro") / np.linalg.norm(H, "fro"), Hl


def drag_feature(W):
    """Quadratic-drag forcing feature phi(w) = |w| w (2-D)."""
    W = np.asarray(W, float)
    return np.linalg.norm(W, axis=1)[:, None] * W


def feature_fit_error(G, phi, S=S_CAL, ddof=1):
    """Fit centred scaled gradients on centred features (least squares) and return
    eps = ||H - S B Sigma_phi B^T S||_F / ||H||_F (in-sample, rank <= q) and the
    leave-one-out cross-validated R^2 of the scaled gradients (mean over components)."""
    Gs = np.asarray(G, float) * S
    phi = np.asarray(phi, float)
    E = len(Gs)
    Gc = Gs - Gs.mean(0)
    P = phi - phi.mean(0)
    B, *_ = np.linalg.lstsq(P, Gc, rcond=None)
    H = Gc.T @ Gc / (E - ddof)
    F = P @ B
    Hf = F.T @ F / (E - ddof)
    eps = np.linalg.norm(H - Hf, "fro") / np.linalg.norm(H, "fro")
    pred = np.zeros_like(Gs)
    for k in range(E):
        m = np.arange(E) != k
        Pm = phi[m] - phi[m].mean(0)
        Bm, *_ = np.linalg.lstsq(Pm, Gs[m] - Gs[m].mean(0), rcond=None)
        pred[k] = Gs[m].mean(0) + (phi[k] - phi[m].mean(0)) @ Bm
    r2 = 1 - ((Gs - pred) ** 2).sum(0) / (Gc ** 2).sum(0)
    return float(eps), r2


def angle(u, v):
    """Angle between eigen-directions (sign invariant), degrees."""
    c = abs(u @ v) / (np.linalg.norm(u) * np.linalg.norm(v))
    return np.degrees(np.arccos(np.clip(c, 0, 1)))


def bootstrap(G, B=2000, seed=0, S=S_CAL):
    rng = np.random.default_rng(seed)
    E = G.shape[0]
    full = summary(G, S)
    out = {k: [] for k in ("fro", "lam1", "lam_hidden", "rho_hidden", "kappa_hidden", "angle_v1")}
    for _ in range(B):
        idx = rng.integers(0, E, E)
        s = summary(G[idx], S)
        for k in ("fro", "lam1", "lam_hidden", "rho_hidden", "kappa_hidden"):
            out[k].append(s[k])
        out["angle_v1"].append(angle(s["V"][:, 0], full["V"][:, 0]))
    res = {}
    for k, v in out.items():
        v = np.asarray(v)
        res[k] = dict(lo=float(np.percentile(v, 2.5)), hi=float(np.percentile(v, 97.5)),
                      median=float(np.median(v)))
    return res


def leave_one_out(G, S=S_CAL):
    full = summary(G, S)["fro"]
    E = G.shape[0]
    return np.array([summary(np.delete(G, k, 0), S)["fro"] / full - 1 for k in range(E)])


def random_subsets(G, sizes=(5, 10, 20, 30, 40), n_draw=1000, seed=0, S=S_CAL):
    rng = np.random.default_rng(seed)
    E = G.shape[0]
    ref = summary(G, S)
    rows = []
    for n in sizes:
        ef, el, ea = [], [], []
        for _ in range(n_draw):
            idx = rng.choice(E, n, replace=False)
            s = summary(G[idx], S)
            ef.append(np.linalg.norm(s["H"] - ref["H"], "fro") / ref["fro"])
            el.append(abs(s["lam1"] - ref["lam1"]) / ref["lam1"])
            ea.append(angle(s["V"][:, 0], ref["V"][:, 0]))
        ef, el, ea = map(np.asarray, (ef, el, ea))
        rows.append(dict(n=n,
                         fro_med=float(np.median(ef)), fro_p5=float(np.percentile(ef, 5)),
                         fro_p95=float(np.percentile(ef, 95)),
                         lam1_med=float(np.median(el)), angle_med=float(np.median(ea)),
                         angle_p95=float(np.percentile(ea, 95))))
    return rows


def scaling_sensitivity(G, n_draw=500, spread=(0.5, 2.0), seed=0, S=S_CAL):
    """Random log-uniform rescalings of every characteristic scale. Directions are
    mapped back to the reference scaled coordinates before comparison."""
    rng = np.random.default_rng(seed)
    ref = summary(G, S)
    ang, rho, kap = [], [], []
    for _ in range(n_draw):
        f = np.exp(rng.uniform(np.log(spread[0]), np.log(spread[1]), len(S)))
        Sa = S * f
        s = summary(G, Sa)
        v = (Sa / S) * s["V"][:, 0]          # map to reference scaled coordinates
        ang.append(angle(v, ref["V"][:, 0]))
        rho.append(s["rho_hidden"])
        kap.append(s["kappa_hidden"] / ref["kappa_hidden"])
    ang, rho, kap = map(np.asarray, (ang, rho, kap))
    return dict(angle_med=float(np.median(ang)), angle_p95=float(np.percentile(ang, 95)),
                rho_range=[float(rho.min()), float(rho.max())],
                kappa_ratio_p5_p95=[float(np.percentile(kap, 5)), float(np.percentile(kap, 95))])
