"""Step 6 - all manuscript figures (600-dpi PNG, IEEE column widths)."""
import os
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cass  # noqa: E402,F401  (checks that the required packages are installed)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.text as mtext
import numpy as np
from matplotlib.patches import Ellipse
from matplotlib.ticker import NullFormatter

from cass import io, model as M

COL1, COL2 = 3.5, 7.16           # IEEE single / double column width (in)
BLUE, VERM, GREEN, PURP = "#0072B2", "#D55E00", "#009E73", "#7A5195"
CASE_STYLE = {"LOW": dict(color=BLUE, marker="o"), "HIGH": dict(color=VERM, marker="s")}
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "STIXGeneral", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "legend.fontsize": 7.5, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "axes.linewidth": 0.6, "grid.linewidth": 0.4, "grid.alpha": 0.35, "lines.linewidth": 1.2,
    "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})


def _latex_available():
    """Typeset with LaTeX (as in the manuscript) only when a TeX installation is present.
    Set CASS_NO_LATEX=1 to force the built-in renderer."""
    if os.environ.get("CASS_NO_LATEX") or not shutil.which("latex"):
        return False
    import io as _io
    try:                                    # a TeX installation may lack packages: test a tiny render
        with plt.rc_context({"text.usetex": True, "text.latex.preamble": _PREAMBLE}):
            f = plt.figure(figsize=(1, 1))
            f.text(0.1, 0.5, r"$\alpha\,\mathbb{R}$ \textbf{x}")
            f.savefig(_io.BytesIO(), format="png")
            plt.close(f)
        return True
    except Exception:                       # noqa: BLE001
        plt.close("all")
        return False


_PREAMBLE = r"\usepackage{amsmath}\usepackage{amssymb}\usepackage{mathptmx}"
USETEX = _latex_available()
if USETEX:
    plt.rcParams.update({"text.usetex": True, "text.latex.preamble": _PREAMBLE})
else:
    print("s06: LaTeX not available - using matplotlib's built-in math renderer. "
          "Figures are equivalent; only the fonts differ slightly from the manuscript.")
    _set_text = mtext.Text.set_text

    def _strip_textbf(t):
        """Remove every \\textbf{...} wrapper (brace-matched); return (text, had_bold)."""
        out, k, bold = [], 0, False
        while True:
            a = t.find("\\textbf{", k)
            if a < 0:
                out.append(t[k:])
                break
            out.append(t[k:a])
            depth, b = 1, a + 8
            while b < len(t) and depth:
                depth += {"{": 1, "}": -1}.get(t[b], 0)
                b += 1
            out.append(t[a + 8:b - 1])
            bold, k = True, b
        return "".join(out), bold

    def _plain_part(t):
        """Convert LaTeX text-mode markup (outside $...$) to plain text."""
        for a, b in (("\\%", "%"), ("---", "\u2014"), ("--", "\u2013"), ("``", "\u201c"), ("''", "\u201d"),
                     ("vs.\\ ", "vs. "), ("\\ ", " "), ("~", " ")):
            t = t.replace(a, b)
        return t

    def _to_mathtext(txt):
        parts = txt.split("$")
        for i in range(len(parts)):
            if i % 2 == 0:
                parts[i] = _plain_part(parts[i])
            else:
                m = parts[i].replace("\\tfrac", "\\frac").replace("\\lVert", "\\|").replace("\\rVert", "\\|")
                m = re.sub(r"\\ge(?![a-zA-Z])", r"\\geq", m)
                parts[i] = re.sub(r"\\le(?![a-zA-Z])", r"\\leq", m)
        return "$".join(parts)

    def _set_text_mathtext(self, s):
        if isinstance(s, str) and ("\\" in s or "--" in s):
            s, bold = _strip_textbf(s)
            if bold:
                if hasattr(self, "_fontproperties"):
                    self.set_fontweight("bold")
                else:                       # called from Text.__init__: apply after construction
                    self._cass_bold = True
            s = _to_mathtext(s)
        return _set_text(self, s)

    _text_init = mtext.Text.__init__

    def _text_init_bold(self, *args, **kwargs):
        _text_init(self, *args, **kwargs)
        if getattr(self, "_cass_bold", False):
            self.set_fontweight("bold")

    mtext.Text.__init__ = _text_init_bold
    mtext.Text.set_text = _set_text_mathtext
FIG = io.FIG
FIG.mkdir(exist_ok=True)


def _save(fig, name, **kw):
    """Write each figure as a 600-dpi PNG (the format used in the manuscript)."""
    fig.savefig(FIG / f"{name}.png", dpi=600, **kw)


def ellipse(ax, mean, cov, color, ls="-", lw=1.0, nstd=2.0, label=None):
    lam, V = np.linalg.eigh(cov)
    lam = np.clip(lam, 0, None)
    ang = np.degrees(np.arctan2(V[1, -1], V[0, -1]))
    e = Ellipse(mean, 2 * nstd * np.sqrt(lam[-1]), 2 * nstd * np.sqrt(lam[0]), angle=ang,
                fill=False, ec=color, ls=ls, lw=lw, label=label)
    ax.add_patch(e)


def fig_winds(P):
    spread = np.load(io.RES / "site_spread.npy")
    inf = io.info()
    z = io.load_winds()
    times = list(z["times"])
    fig, axs = plt.subplots(1, 2, figsize=(COL2, 2.9), gridspec_kw=dict(width_ratios=[1, 1.6]))
    ax = axs[0]
    for case in ("LOW", "HIGH"):
        W = np.asarray(z["site"][times.index(P[case]["time"])])
        st = CASE_STYLE[case]
        ax.scatter(W[:, 0], W[:, 1], s=10, alpha=0.8, color=st["color"], marker=st["marker"],
                   lw=0, label=f"{case} members")
        ellipse(ax, W.mean(0), np.cov(W.T), st["color"], label=f"{case} $2\\sigma$ ellipse")
        ax.plot(*W.mean(0), marker="+", color="k", ms=7, mew=1.2)
    ax.set_xlabel(r"$u_{10}$ ($\mathrm{m\,s^{-1}}$)")
    ax.set_ylabel(r"$v_{10}$ ($\mathrm{m\,s^{-1}}$)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, frameon=False, handlelength=1.2,
              fontsize=6.8)
    ax.set_title(r"(a) Member winds at $28.0^\circ\mathrm{N}$, $95.5^\circ\mathrm{E}$")
    ax = axs[1]
    t = np.arange(len(spread))
    ax.plot(t, spread, color="0.35", lw=1.0, marker="o", ms=2.5)
    for case in ("LOW", "HIGH"):
        k = times.index(P[case]["time"])
        st = CASE_STYLE[case]
        ax.plot(k, spread[k], ls="none", ms=7, mfc="none", mew=1.4, **st)
        ax.annotate(case, (k, spread[k]), xytext=(5, 4 if case == "HIGH" else -10),
                    textcoords="offset points", fontsize=7.5)
    ax.set_xlabel(r"Analysis time index ($00/12$ UTC, $1$--$31$ July $2022$)")
    ax.set_ylabel(r"Spread $s$ ($\mathrm{m\,s^{-1}}$)")
    ax.grid(True)
    ax.set_title(f"(b) Ensemble spread $s$ at all ${inf['n_times']}$ times")
    fig.tight_layout()
    _save(fig, "fig_winds")
    plt.close(fig)


def fig_gradient_cloud(P):
    fig, axs = plt.subplots(1, 2, figsize=(COL2, 2.7))
    for ax, case in zip(axs, ("LOW", "HIGH")):
        g = np.load(io.ROOT / f"data/processed/gradients_{case}.npz")
        G = g["G"] * M.S_CAL                        # scaled gradients S g_k
        p = P[case]
        V = np.asarray(p["V"])[:, :2]
        c = np.asarray(p["c"])
        Z = (G - G.mean(0)) @ V                    # centred, in the dominant plane
        st = CASE_STYLE[case]
        ax.scatter(Z[:, 0], Z[:, 1], s=10, color=st["color"], marker=st["marker"], lw=0, alpha=0.85,
                   label="members $S^{\\top}(g_k-\\bar g)$")
        H = np.asarray(p["H"])
        Hl = np.asarray(p["H_lin"])
        ellipse(ax, (0, 0), V.T @ H @ V, "k", lw=1.1, label="$H$ (sample), $2\\sigma$")
        ellipse(ax, (0, 0), V.T @ Hl @ V, GREEN, ls="--", lw=1.3,
                label="tangent $S^{\\top}K\\Sigma_wK^{\\top}S$, $2\\sigma$")
        ellipse(ax, (0, 0), V.T @ np.asarray(p["H_drag"]) @ V, VERM if case == "LOW" else BLUE, ls="-.",
                lw=1.3, label="drag feature $\\|w\\|w$, $2\\sigma$")
        # mean-gradient nullspace inside this plane: directions y with (V y) . c = 0
        cp = V.T @ c
        nd = np.array([-cp[1], cp[0]])
        nd /= np.linalg.norm(nd)
        L = 1.1 * np.abs(Z).max()
        ax.plot([-L * nd[0], L * nd[0]], [-L * nd[1], L * nd[1]], color=PURP, lw=1.0, ls=":",
                label=r"$c^{\top}x=0$ (in plane)")
        ax.set_xlabel(r"coordinate on $v_1$")
        ax.set_ylabel(r"coordinate on $v_2$")
        lim = 1.15 * max(np.abs(Z).max(), 2.2 * np.sqrt(np.linalg.eigvalsh(V.T @ H @ V)).max())
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-0.6 * lim, 0.6 * lim)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True)
        ax.set_title(f"({'a' if case == 'LOW' else 'b'}) {case}: $\\lambda_1={p['lam1']:.3g}$, "
                     f"$\\lambda_2={p['lam2']:.3g}$")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, fontsize=7, bbox_to_anchor=(0.5, -0.03))
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    _save(fig, "fig_gradient_cloud")
    plt.close(fig)


def fig_verification():
    v = io.load("verification.json")
    fig, axs = plt.subplots(1, 2, figsize=(COL2, 2.4))
    ax = axs[0]
    for key, tt in v["taylor"].items():
        case = key.split("_")[0]
        st = CASE_STYLE[case]
        h = np.asarray(tt["h"])
        ax.loglog(h, tt["r0"], color=st["color"], lw=0.6, alpha=0.5)
        ax.loglog(h, tt["r1"], color=st["color"], lw=0.8, marker=st["marker"], ms=2.5)
    h = np.asarray(next(iter(v["taylor"].values()))["h"])
    r1 = np.asarray(next(iter(v["taylor"].values()))["r1"])
    ax.loglog(h, r1[0] * (h / h[0]) ** 2, "k--", lw=0.8, label=r"slope $2$")
    ax.loglog(h, 0.5 * np.asarray(next(iter(v["taylor"].values()))["r0"])[0] * (h / h[0]), "k:",
              lw=0.8, label=r"slope $1$")
    ax.plot([], [], color=BLUE, marker="o", ms=3, label="LOW members")
    ax.plot([], [], color=VERM, marker="s", ms=3, label="HIGH members")
    ax.set_xlabel("$h$")
    ax.set_ylabel(r"remainder $r_0(h)$, $r_1(h)$")
    ax.grid(True, which="major")
    ax.legend(frameon=False, fontsize=6.8)
    ax.set_title(r"(a) Taylor test: $r_1$ (thick), $r_0$ (thin)")
    ax = axs[1]
    for case in ("LOW", "HIGH"):
        st = CASE_STYLE[case]
        ca = v["continuous_adjoint"][case]
        n = np.array([int(k) for k in ca])
        e = np.array([ca[k]["max_rel_vs_discrete"] for k in ca])
        ax.loglog(n, e, color=st["color"], marker=st["marker"], ms=4, label=f"{case}: nodes $n_q$")
        dr = v["dt_refinement"][case]
        ax.loglog(dr["steps"], dr["max_rel_err"], color=st["color"], marker=st["marker"], ms=4,
                  ls="--", mfc="none", label=f"{case}: steps $N$")
    ax.loglog(n, e[0] * (n / n[0]) ** -2.0, "k:", lw=0.8, label="slope $-2$")
    ax.set_xlabel(r"quadrature nodes $n_q$ / RK4 steps $N$")
    ax.set_ylabel(r"$\max_i |a_i-f_i|/|f_i|$")
    ax.set_xticks([500, 1000, 2000, 4000])
    ax.set_xticklabels([r"$500$", r"$1000$", r"$2000$", r"$4000$"])
    ax.minorticks_off()
    ax.grid(True, which="major")
    ax.legend(frameon=False, fontsize=6.3, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.25))
    ax.set_title("(b) Discretization refinement")
    fig.tight_layout()
    fig.set_size_inches(COL2, 2.9)
    _save(fig, "fig_verification")
    plt.close(fig)


def fig_multicase(P):
    z = np.load(io.RES / "multicase.npz", allow_pickle=True)
    A, cols = z["A"], list(z["cols"])
    col = lambda n: A[:, cols.index(n)]
    site_nb = A[:, 1] == 0
    fig, axs = plt.subplots(1, 2, figsize=(COL2, 2.5))
    ax = axs[0]
    ax.loglog(col("trSw")[site_nb], col("lam1")[site_nb], "o", ms=2.2, color=BLUE, alpha=0.45, mew=0,
              label="site $5\\times5$ neighborhood")
    ax.loglog(col("trSw")[~site_nb], col("lam1")[~site_nb], "^", ms=2.6, color=GREEN, alpha=0.6, mew=0,
              label="contrasting regimes")
    for case in ("LOW", "HIGH"):
        st = CASE_STYLE[case]
        ax.plot(np.trace(np.asarray(P[case]["Sigma_w"])), P[case]["lam1"], ls="none", ms=7,
                mfc="none", mew=1.4, label=case, **st)
    ax.set_xlabel(r"$\mathrm{tr}\,\Sigma_w$ ($\mathrm{m^2\,s^{-2}}$)")
    ax.set_ylabel(r"$\lambda_1(H)$")
    ax.grid(True, which="major")
    ax.legend(frameon=False, fontsize=6.6, loc="upper left")
    ax.set_title(f"(a) $\\lambda_1(H)$ versus $\\mathrm{{tr}}\\,\\Sigma_w$ ($n={len(A)}$)")
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax = axs[1]
    r = col("spread_ratio")
    ax.loglog(r, col("prop2_err"), "o", ms=2.0, color=GREEN, alpha=0.4, mew=0,
              label=r"tangent $\epsilon_{\mathrm{lin}}$ (no fit)")
    ax.loglog(r, col("eps_linfit"), "^", ms=2.0, color=PURP, alpha=0.4, mew=0,
              label=r"best linear in $w$, $\epsilon_w$")
    ax.loglog(r, col("eps_drag"), "s", ms=2.0, color="0.25", alpha=0.4, mew=0,
              label=r"drag feature, $\epsilon_{\mathrm{drag}}$")
    for case in ("LOW", "HIGH"):
        st = CASE_STYLE[case]
        ax.plot(P[case]["spread_ratio"], P[case]["prop2_rel_err"], ls="none", ms=7, mfc="none", mew=1.4,
                label=case, **st)
    ax.set_xlabel(r"relative spread $s/\|\bar w\|$")
    ax.set_ylabel(r"relative Frobenius error of $H$")
    ax.grid(True, which="major")
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.legend(frameon=False, fontsize=6.3, loc="lower right")
    ax.set_title("(b) Structure of $H$ vs. forcing representation")
    fig.tight_layout()
    _save(fig, "fig_multicase")
    plt.close(fig)


def fig_subsets_alpha(P):
    fig, axs = plt.subplots(1, 2, figsize=(COL2, 2.4))
    ax = axs[0]
    for case in ("LOW", "HIGH"):
        st = CASE_STYLE[case]
        rows = P[case]["subsets"]
        n = [r["n"] for r in rows]
        ax.plot(n, [100 * r["fro_med"] for r in rows], label=f"{case} median", **st, ms=4)
        ax.fill_between(n, [100 * r["fro_p5"] for r in rows], [100 * r["fro_p95"] for r in rows],
                        color=st["color"], alpha=0.15, lw=0)
    ax.set_xlabel(r"random subset size $n$ (of $50$)")
    ax.set_ylabel(r"relative Frobenius error of $H$ (\%)")
    ax.set_yscale("log")
    ax.grid(True, which="major")
    ax.legend(frameon=False)
    ax.set_title(r"(a) Subset sensitivity (band: $5$--$95\%$)")
    ax = axs[1]
    for case in ("LOW", "HIGH"):
        st = CASE_STYLE[case]
        rows = P[case]["alpha_sweep"]["box"]["rows"]
        a = np.array([r["alpha"] for r in rows])
        ax.loglog(a, [r["err_mean"] for r in rows], label=f"{case}: mean change", **st, ms=3.5)
        ax.loglog(a, [r["err_std"] for r in rows], ls="--", mfc="none", color=st["color"],
                  marker=st["marker"], ms=3.5, label=f"{case}: dispersion")
    ax.loglog(a, [r["err_mean"] for r in rows][0] * (a / a[0]) ** 2, "k:", lw=0.8, label=r"slope $2$")
    ax.set_xlabel(r"step scale $\alpha$")
    ax.set_ylabel(r"$|\Delta_{\mathrm{nl}}-\Delta_{\mathrm{lin}}|$")
    ax.grid(True, which="major")
    ax.legend(frameon=False, fontsize=6.4)
    ax.set_title(r"(b) First-order accuracy, $x=-\mathrm{sign}(c)$")
    fig.tight_layout()
    _save(fig, "fig_subsets_alpha")
    plt.close(fig)


RULES = ["mean-box", "mean-dir", "CASS-1 (b=1.645)", "CASS-1 (b=2b*)", "CASS-1 (b=5b*)",
         "mean-2", "CASS-2 (b=1.645)", "SAA", "CVaR-0.9"]
LABELS = {"mean-box": r"Mean, box ($\beta=0$)", "mean-dir": "Mean, direction",
          "CASS-1 (b=1.645)": r"CASS-1, $\beta=1.645$", "CASS-1 (b=2b*)": r"CASS-1, $\beta=2\beta^*$",
          "CASS-1 (b=5b*)": r"CASS-1, $\beta=5\beta^*$", "mean-2": "Mean, 2nd order",
          "CASS-2 (b=1.645)": r"CASS-2, $\beta=1.645$", "SAA": "SAA (nonlinear)",
          "CVaR-0.9": r"$\mathrm{CVaR}_{0.9}$, 2nd order"}


def fig_control():
    cp = io.load("control_primary.json")
    ca = io.load("control_alltimes.json")["aggregate"]
    rules = [r for r in RULES if r in cp["LOW"]["summary"]]
    y = np.arange(len(rules))
    fig, axs = plt.subplots(1, 2, figsize=(COL2, 2.9), sharey=True)
    for ax, key, title in ((axs[0], "d_mean", r"(a) $\Delta\bar J$ on held-out members"),
                           (axs[1], "d_p95", r"(b) $\Delta J_{95}$ on held-out members")):
        h = 0.26
        for off, case in ((-h, "LOW"), (0, "HIGH")):
            st = CASE_STYLE[case]
            vals = [cp[case]["summary"][r][key] for r in rules]
            ax.barh(y + off, vals, height=h * 0.92, color=st["color"], label=f"{case} ($100$ splits)")
        vals = [ca[r][key] if r in ca else np.nan for r in rules]
        ax.barh(y + h, vals, height=h * 0.92, color="0.55", label=r"all $62$ times ($10$ splits each)")
        ax.axvline(0, color="k", lw=0.6)
        ax.grid(True, axis="x")
        ax.set_title(title)
        ax.set_xlabel(r"change relative to $u_0$")
    axs[0].set_yticks(y)
    axs[0].set_yticklabels([LABELS[r] for r in rules])
    axs[0].invert_yaxis()
    h, l = axs[1].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, frameon=False, fontsize=7, bbox_to_anchor=(0.5, 1.04))
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _save(fig, "fig_control")
    plt.close(fig)


ICON_FONT = io.ROOT / "assets/fonts/fa-solid-900.otf"   # Font Awesome Free (SIL OFL 1.1)
ICON = {"satellite": "", "cloud": "", "wind": "", "chip": "", "laptop": "",
        "chart": "", "sqrt": "", "scale": "", "route": "", "glass": "",
        "check": "", "target": "", "line": "", "box": "", "tick": "", "layers": "", "earth": "", "gears": "", "gamepad": "",
        "trophy": "", "rotate": "", "userminus": "", "expand": "", "bulb": "",
        "sliders": "", "ccheck": "", "mapmarker": ""}


def _icon(ax, x, y, name, color, size=17):
    """Draw an icon glyph as a filled vector path (no icon font is needed to view the figure)."""
    import matplotlib.transforms as mt
    from matplotlib.font_manager import FontProperties
    from matplotlib.patches import PathPatch
    from matplotlib.textpath import TextPath
    fp = FontProperties(fname=str(ICON_FONT))
    path = TextPath((0, 0), ICON[name], size=size, prop=fp, usetex=False)
    v = path.vertices
    cx, cy = (v[:, 0].min() + v[:, 0].max()) / 2, (v[:, 1].min() + v[:, 1].max()) / 2
    # points -> inches -> display, then shifted to the display position of (x, y) (evaluated lazily)
    tr = (mt.Affine2D().translate(-cx, -cy).scale(1 / 72) + ax.figure.dpi_scale_trans
          + mt.ScaledTranslation(x, y, ax.transData))
    ax.add_patch(PathPatch(path, transform=tr, fc=color, ec="none", zorder=5))


def _drone(ax, x, y, s, color, k):
    """Quadrotor icon: X-frame, four rotors and a hanging parcel (k = fig width / fig height)."""
    from matplotlib.patches import Rectangle, Ellipse as El
    for dx, dy in ((-1, -1), (1, -1), (-1, 1), (1, 1)):
        ex, ey = x + dx * 0.8 * s, y + dy * 0.45 * s * k
        ax.plot([x, ex], [y, ey], color=color, lw=1.6, solid_capstyle="round", zorder=5)
        ax.add_patch(El((ex, ey + 0.12 * s * k), 0.95 * s, 0.24 * s * k, fc="none", ec=color, lw=1.0, zorder=5))
        ax.plot([ex, ex], [ey, ey + 0.12 * s * k], color=color, lw=1.1, zorder=5)
    ax.add_patch(Rectangle((x - 0.3 * s, y - 0.18 * s * k), 0.6 * s, 0.36 * s * k, fc=color, ec=color, zorder=5))
    ax.plot([x, x], [y - 0.18 * s * k, y - 0.62 * s * k], color=color, lw=0.8, zorder=5)
    ax.add_patch(Rectangle((x - 0.24 * s, y - 0.95 * s * k), 0.48 * s, 0.33 * s * k, fc=color, ec=color, zorder=5))


def _macros():
    import re
    txt = (io.ROOT / "tables/numbers.tex").read_text(encoding="utf-8")
    return dict(re.findall(r"^\\newcommand\{\\(\w+)\}\{(.*)\}\s*$", txt, flags=re.M))


def fig_overview():
    """Framework of the study (infographic with icons and schematic diagrams only)."""
    from matplotlib.patches import FancyBboxPatch, Rectangle, Circle, FancyArrowPatch, Ellipse as El
    N = _macros()
    mc = io.load("multicase.json")["all"]
    ctl = io.load("control_alltimes.json")["aggregate"]

    W, H = COL2, 6.3
    BIG, BODY, SMALL, HEAD = 7.6, 7.2, 6.8, 8.8       # font sizes (pt)
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1], zorder=0)
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    T = lambda x, y, t, fs=BODY, **kw: ax.text(x, y, t, fontsize=fs, **{"ha": "left", "va": "center", **kw})
    GREY = "0.35"
    TEAL, CRIM = "#2E7D8C", "#B5174B"

    mx, g = 0.03, 0.25
    pw = (W - 2 * mx - 2 * g) / 3
    rows = {0: (3.38, 2.89), 1: (0.03, 2.97)}
    hh = 0.33
    panels = []

    def panel(col, row, color, num, title, icon):
        x = mx + col * (pw + g)
        y, h = rows[row]
        panels.append((x, y, pw, h))
        ax.add_patch(FancyBboxPatch((x, y), pw, h, boxstyle="round,pad=0,rounding_size=0.08",
                                    fc=matplotlib.colors.to_rgba(color, 0.045), ec=color, lw=1.1))
        ax.add_patch(FancyBboxPatch((x, y + h - hh), pw, hh, boxstyle="round,pad=0,rounding_size=0.08",
                                    fc=color, ec=color, lw=1.1))
        ax.add_patch(Rectangle((x, y + h - hh), pw, hh / 2, fc=color, ec="none"))
        cy = y + h - hh / 2
        ax.add_patch(Circle((x + 0.17, cy), 0.105, fc="white", ec="none", zorder=4))
        T(x + 0.17, cy, rf"\textbf{{{num}}}", 8.2, ha="center", color=color, zorder=6)
        _icon(ax, x + 0.41, cy, icon, "white", size=10.5)
        T(x + 0.56, cy, rf"\textbf{{{title}}}", HEAD, color="white")
        return x, y, h, y + h - hh

    def tile(x, y, color, icon, s=0.34, size=13):
        ax.add_patch(FancyBboxPatch((x - s / 2, y - s / 2), s, s, boxstyle="round,pad=0,rounding_size=0.06",
                                    fc=color, ec="none", zorder=3))
        if icon == "drone":
            _drone(ax, x, y + 0.025, 0.07, "white", 1.0)
        else:
            _icon(ax, x, y, icon, "white", size=size)

    def arrow(p0, p1, color="0.35", lw=1.0, ms=8, **kw):
        ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=ms, lw=lw, color=color,
                                     shrinkA=0, shrinkB=0, **kw))

    def bullets(x, y0, items, dy=0.2, fs=BODY):
        for n, t in enumerate(items):
            T(x, y0 - n * dy, r"$\bullet$ " + t, fs)

    # ===== 1 weather ensemble =====================================================================
    x, y, h, hb = panel(0, 0, BLUE, 1, "Weather ensemble and data", "earth")
    xs = [x + 0.33 + j * 0.52 for j in range(4)]
    yi = hb - 0.3
    _icon(ax, xs[0], yi, "satellite", BLUE, size=16)
    _icon(ax, xs[1], yi, "cloud", BLUE, size=16)
    _icon(ax, xs[2], yi, "wind", BLUE, size=16)
    _drone(ax, xs[3], yi + 0.01, 0.09, BLUE, 1.0)
    for j in range(3):
        arrow((xs[j] + 0.17, yi), (xs[j + 1] - 0.17, yi), color="0.55", lw=0.8, ms=6)
    for xx, t in zip(xs, [r"ECMWF ENS", rf"$E={N['NMembers']}$", r"wind $w_k$", r"quadrotor"]):
        T(xx, yi - 0.27, t, SMALL, ha="center", color=GREY)
    by0, bh = y + 0.8, 0.82
    for bx0, ttl in ((x + 0.12, r"$5\times5$ neighborhood"), (x + 1.14, rf"${N['NMembers']}$ members")):
        ax.add_patch(FancyBboxPatch((bx0, by0), 0.94, bh, boxstyle="round,pad=0,rounding_size=0.05",
                                    fc="white", ec=BLUE, lw=0.7))
        T(bx0 + 0.47, by0 + bh + 0.11, ttl, SMALL, ha="center", color="0.2")
    cx, cy = x + 0.59, by0 + 0.47
    for i_ in range(5):
        for j_ in range(5):
            ax.add_patch(Circle((cx + (i_ - 2) * 0.1, cy + (j_ - 2) * 0.075), 0.018, fc=BLUE, ec="none", zorder=3))
    _icon(ax, cx, cy + 0.06, "mapmarker", CRIM, size=11)
    T(cx, by0 + 0.1, r"$25$ grid points", SMALL, ha="center", color=GREY)
    fx, fy = x + 1.61, by0 + 0.48
    for ang, L in zip(np.linspace(0, 2 * np.pi, 13)[:-1], [0.27, 0.2, 0.25, 0.17, 0.28, 0.22, 0.19, 0.26,
                                                         0.21, 0.23, 0.18, 0.24]):
        arrow((fx, fy), (fx + L * np.cos(ang), fy + 0.85 * L * np.sin(ang)), color=BLUE, lw=0.8, ms=5)
    ax.add_patch(Circle((fx, fy), 0.028, fc="k", ec="none", zorder=4))
    T(fx, by0 + 0.1, r"$w_1,\dots,w_{50}$", SMALL, ha="center", color=GREY)
    T(x + 0.12, y + 0.57, r"TIGGE, July $2022$, $00/12$ UTC", BODY)
    T(x + 0.12, y + 0.36, r"site $28.0^\circ\mathrm{N}$, $95.5^\circ\mathrm{E}$; $7$ regimes", BODY)
    T(x + 0.12, y + 0.15, rf"LOW $s={N['LowSpread']}$, HIGH $s={N['HighSpread']}\ \mathrm{{m\,s^{{-1}}}}$", BODY)

    # ===== 2 quadrotor model ======================================================================
    x, y, h, hb = panel(1, 0, GREEN, 2, "Quadrotor model", "gears")
    items = [("drone", r"6-DOF rigid body", r"state $(p,v,R,\omega)$, $T_f=2\ \mathrm{s}$"),
             ("gamepad", r"Control $u=(T,\tau_x,\tau_y,\tau_z)$", r"near-hover command $u_0$"),
             ("wind", r"Quadratic drag", r"$F_d\propto-\|v-w_k\|\,(v-w_k)$"),
             ("target", r"Objective $J_k(u)$", r"terminal tracking + regularization")]
    for n, (ic, t1, t2) in enumerate(items):
        yy = hb - 0.33 - n * 0.44
        tile(x + 0.26, yy, GREEN, ic)
        T(x + 0.52, yy + 0.08, rf"\textbf{{{t1}}}", BIG)
        T(x + 0.52, yy - 0.1, t2, SMALL, color=GREY)
    ax.add_patch(FancyBboxPatch((x + 0.08, y + 0.08), pw - 0.16, 0.52, boxstyle="round,pad=0,rounding_size=0.05",
                                fc="white", ec=GREEN, lw=0.8, ls="--"))
    _icon(ax, x + 0.26, y + 0.34, "sliders", GREEN, size=13)
    T(x + 0.52, y + 0.43, r"\textbf{Model variants}", BIG)
    T(x + 0.52, y + 0.24, r"rotor drag, closed loop, $u^*$", SMALL, color=GREY)

    # ===== 3 exact gradients ======================================================================
    x, y, h, hb = panel(2, 0, PURP, 3, "Exact member-wise gradients", "chip")
    y1 = hb - 0.5
    for xx, t in ((x + 0.3, r"$u$"), (x + pw - 0.3, r"$J_k$")):
        ax.add_patch(Circle((xx, y1), 0.15, fc="white", ec=PURP, lw=1.1, zorder=3))
        T(xx, y1, t, 8.5, ha="center", zorder=4)
    arrow((x + 0.48, y1 + 0.055), (x + pw - 0.48, y1 + 0.055), color="0.4", lw=1.0, ms=7)
    arrow((x + pw - 0.48, y1 - 0.055), (x + 0.48, y1 - 0.055), color=PURP, lw=1.2, ms=7)
    T(x + pw / 2, y1 + 0.2, r"forward RK4, $N=2000$", SMALL, ha="center", color=GREY)
    T(x + pw / 2, y1 - 0.2, r"reverse AD (discrete adjoint)", SMALL, ha="center", color=PURP)
    T(x + pw / 2, hb - 0.98, r"$g_k=\nabla_u J_k(u_0)$,\ $\nabla_u^2J_k$,\ $K=\tfrac{\partial g}{\partial w}$", BODY,
      ha="center")
    tile(x + 0.5, y + 0.95, PURP, "glass", s=0.62, size=23)
    T(x + 0.5, y + 0.5, r"\textbf{Verification}", BIG, ha="center", color=PURP)
    for n, t in enumerate([rf"Taylor order ${N['TaylorOrderRange']}$", r"finite differences",
                           r"step refinement", r"continuous adjoint"]):
        yy = y + 1.27 - n * 0.24
        _icon(ax, x + 1.03, yy, "ccheck", GREEN, size=8.5)
        T(x + 1.14, yy, t, BODY)
    T(x + pw / 2, y + 0.18, r"per case: $E=50$ gradients $g_k\in\mathbb{R}^4$", BODY, ha="center")

    # arrows between panels and connector
    for c in range(2):
        xa = mx + c * (pw + g) + pw
        arrow((xa + 0.03, 3.38 + 1.45), (xa + g - 0.03, 3.38 + 1.45), color="0.35", lw=1.4, ms=11)
        arrow((xa + 0.03, 0.03 + 1.5), (xa + g - 0.03, 0.03 + 1.5), color="0.35", lw=1.4, ms=11)
    x3, x4, ym = mx + 2 * (pw + g) + pw / 2, mx + pw / 2, (3.0 + 3.38) / 2
    ax.plot([x3, x3, x4], [3.38, ym, ym], color="0.35", lw=1.2, solid_joinstyle="round")
    arrow((x4, ym), (x4, 3.005), color="0.35", lw=1.2, ms=9)
    ax.text(W / 2, ym, r"member-wise gradients $\{g_k\}_{k=1}^{50}$ for each of $" + N["NCases"].replace(",", "{,}")
            + r"$ cases", fontsize=BODY + 0.2, ha="center", va="center", zorder=6,
            bbox=dict(boxstyle="round,pad=0.28,rounding_size=0.3", fc="white", ec="0.35", lw=0.8))

    # ===== 4 CASS statistics ======================================================================
    x, y, h, hb = panel(0, 1, VERM, 4, "CASS statistics", "chart")
    T(x + pw / 2, hb - 0.22, r"$\bar g=\tfrac{1}{E}\sum_k g_k$,\ \ $c=S^{\top}\bar g$", BODY, ha="center")
    T(x + pw / 2, hb - 0.47, r"$D=\tfrac{1}{E-1}\sum_k(g_k-\bar g)(g_k-\bar g)^{\top}$", BODY, ha="center")
    T(x + pw / 2, hb - 0.72, r"$H=S^{\top}DS$,\ \ $S=\mathrm{diag}(mg,\,I_B\alpha_{\mathrm{ref}})$", BODY, ha="center")
    ax.add_patch(FancyBboxPatch((x + 0.1, y + 1.0), pw - 0.2, 0.78, boxstyle="round,pad=0,rounding_size=0.05",
                                fc="white", ec=VERM, lw=0.7))
    ex, ey, th = x + 0.55, y + 1.39, 20.0
    ax.add_patch(El((ex, ey), 0.74, 0.38, angle=th, fc=matplotlib.colors.to_rgba(VERM, 0.12), ec="k", lw=0.9))
    for ang, L in zip(np.radians(th + np.array([-12, 8, 170, 195, 175, 15, 60, 245, 0, 185])),
                      [0.34, 0.28, 0.32, 0.26, 0.21, 0.19, 0.12, 0.12, 0.25, 0.29]):
        arrow((ex, ey), (ex + L * np.cos(ang), ey + L * np.sin(ang)), color=VERM, lw=0.8, ms=5)
    hd = np.radians(th + 75)
    ax.plot([ex - 0.25 * np.cos(hd), ex + 0.25 * np.cos(hd)], [ey - 0.25 * np.sin(hd), ey + 0.25 * np.sin(hd)],
            ":", color=PURP, lw=1.1)
    for n, (cc, t) in enumerate([(VERM, r"$g_k-\bar g$"), ("k", r"$2\sigma$ ellipse of $H$"),
                                  (PURP, r"hidden: $c^{\top}x=0$"), (GREEN, r"$\lambda_1\ge\lambda_2\gg\lambda_{3,4}$")]):
        yy = y + 1.66 - n * 0.18
        ax.add_patch(Circle((x + 1.08, yy), 0.03, fc=cc, ec="none"))
        T(x + 1.15, yy, t, SMALL)
    T(x + 0.12, y + 0.86, r"\textbf{Proved properties}", BIG, color=VERM)
    bullets(x + 0.14, y + 0.66, [r"Prop.~1: $\lambda_2\le\lambda_{\mathrm{hid}}\le\lambda_1$",
                                 r"Prop.~2: leading term $K\Sigma_wK^{\top}$, rank $\le2$",
                                 r"Remark~2: drag, $D\approx B\Sigma_{\phi}B^{\top}$",
                                 r"Prop.~3: CASS-1 $=$ mean if $\beta<\beta^*$"], dy=0.18)

    # ===== 5 decisions ============================================================================
    x, y, h, hb = panel(1, 1, TEAL, 5, "Out-of-sample decisions", "scale")
    T(x + pw / 2, hb - 0.22, r"$\min_{\|x\|_\infty\le1}\ c^{\top}x+\beta\,\sigma(x)$", BODY, ha="center")
    T(x + pw / 2, hb - 0.47, r"$\sigma(x)=\sqrt{x^{\top}Hx}$", BODY, ha="center")
    T(x + pw / 2, hb - 0.69, r"$\Delta J_{95}$ vs.\ $u_0$ on held-out members", SMALL, ha="center", color=GREY)
    v2 = ctl["mean-2"]["d_p95"]
    v1 = sorted(ctl[r]["d_p95"] for r in ("mean-box", "mean-dir", "CASS-1 (b=1.645)"))

    def chips(yy, labels, color, fill):
        cg = 0.05
        cw = (pw - 0.2 - (len(labels) - 1) * cg) / len(labels)
        for n, lab in enumerate(labels):
            x_ = x + 0.1 + n * (cw + cg)
            ax.add_patch(FancyBboxPatch((x_, yy - 0.11), cw, 0.22, boxstyle="round,pad=0,rounding_size=0.07",
                                        fc=fill, ec=color, lw=0.9))
            T(x_ + cw / 2, yy, lab, BODY, ha="center")
    _icon(ax, x + 0.2, hb - 0.95, "ccheck", GREEN, size=9.5)
    T(x + 0.34, hb - 0.95, rf"\textbf{{Curvature-aware}}\ \ $\Delta J_{{95}}\approx{v2:.3f}$", BIG, color=GREEN)
    chips(hb - 1.22, [r"Mean-2", r"CASS-2", r"SAA", r"$\mathrm{CVaR}_{0.9}$"], GREEN,
          matplotlib.colors.to_rgba(GREEN, 0.14))
    _icon(ax, x + 0.2, hb - 1.55, "layers", "0.45", size=9.5)
    T(x + 0.34, hb - 1.55, rf"\textbf{{First-order}}\ \ $\Delta J_{{95}}\approx{v1[-1]:.2f}$", BIG, color="0.3")
    chips(hb - 1.82, [r"Mean", r"Mean-dir", r"CASS-1"], "0.5", "0.93")
    _icon(ax, x + 0.22, y + 0.5, "trophy", "#C9A227", size=14)
    T(x + 0.42, y + 0.59, r"Mean-2 $\approx$ CASS-2 $\approx$ SAA", BODY)
    T(x + 0.42, y + 0.4, r"$\succ$ Mean-dir $\succ$ CASS-1 $\approx$ Mean", BODY)
    T(x + pw / 2, y + 0.15, r"$25$ training / $25$ test members; all $62$ analysis times", SMALL, ha="center", color=GREY)

    # ===== 6 what the study shows =================================================================
    x, y, h, hb = panel(2, 1, CRIM, 6, "What the study shows", "bulb")
    _icon(ax, x + 0.2, hb - 0.24, "layers", CRIM, size=9.5)
    T(x + 0.34, hb - 0.24, r"\textbf{Structure of the dispersion}", BIG, color=CRIM)
    bullets(x + 0.14, hb - 0.46, [
        rf"effective rank $2$ in all ${N['NCases'].replace(',', '{,}')}$ cases",
        rf"drag feature $\|w\|w$, median $\epsilon={mc['eps_drag_median']:.3f}$",
        r"tangent $K\Sigma_wK^{\top}$ only if $s/\|\bar w\|<0.2$",
        rf"$\lambda_1$ tracks $\mathrm{{tr}}\,\Sigma_w$ ($\rho_s={N['MCAllRhoLamTr']}$)",
        rf"$\lambda_1$ tracks $\mathrm{{std}}\,J$ ($\rho_s={N['MCAllRhoJstdLam']}$)"], dy=0.19)
    yB = hb - 1.5
    _icon(ax, x + 0.2, yB, "ccheck", GREEN, size=9.5)
    T(x + 0.34, yB, r"\textbf{Decisions}", BIG, color=GREEN)
    bullets(x + 0.14, yB - 0.23, [r"penalty inactive below $\beta^*$",
                                  r"curvature, not dispersion, helps",
                                  rf"hidden: median $\kappa_{{\mathrm{{hid}}}}={mc['kappa_hidden_median']:.3f}$"], dy=0.19)
    ax.plot([x + 0.1, x + pw - 0.1], [y + 0.36, y + 0.36], color="0.75", lw=0.6)
    rr = fig.canvas.get_renderer()
    tags = [("rotate", "Bootstrap"), ("userminus", "LOO"), ("layers", "Subsets"), ("expand", "Scaling")]
    widths = []
    for ic, t in tags:
        tt = T(0, -5, t, SMALL)
        widths.append(tt.get_window_extent(rr).width / fig.dpi + 0.17)
        tt.remove()
    gap_ = (pw - 0.2 - sum(widths)) / (len(tags) - 1)
    xx = x + 0.1
    for (ic, t), wd in zip(tags, widths):
        _icon(ax, xx + 0.06, y + 0.18, ic, CRIM, size=7.5)
        T(xx + 0.16, y + 0.18, t, SMALL)
        xx += wd + gap_

    # ---- overflow check: every text must stay inside its panel ----------------------------------
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    bad = 0
    for t in ax.texts:
        bb = t.get_window_extent(r)
        x0, x1 = bb.x0 / fig.dpi, bb.x1 / fig.dpi
        tx, ty = t.get_position()
        for (px, py, pw_, ph) in panels:
            if px <= tx <= px + pw_ and py <= ty <= py + ph:
                if x0 < px + 0.03 or x1 > px + pw_ - 0.03:
                    bad += 1
                    print(f"  overflow: {t.get_text()[:60]!r} [{x0 - px:.2f}, {x1 - px:.2f}] of {pw_:.2f} in")
    if bad == 0:
        print("  fig_overview: all text inside panels")
    _save(fig, "fig_overview", bbox_inches=None)
    plt.close(fig)


def main():
    fig_overview()
    P = io.load("primary.json")
    fig_winds(P)
    fig_gradient_cloud(P)
    fig_verification()
    fig_multicase(P)
    fig_subsets_alpha(P)
    fig_control()
    print("figures:", sorted(p.name for p in FIG.glob("*.png")))


if __name__ == "__main__":
    main()
