"""Step 7 - write every number and table used in the manuscript as LaTeX.

Outputs (tables/):
  numbers.tex          \\newcommand macros used in the text
  tab_verification.tex tab_primary.tex tab_control.tex tab_variants.tex
  tab_supp_raw.tex     (supplementary: raw-coordinate values)
  tab_locations.tex    (appendix: coordinates of the 32 locations)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cass  # noqa: E402,F401  (checks that the required packages are installed)
import numpy as np

from cass import io, model as M

OUT = io.ROOT / "tables"
OUT.mkdir(parents=True, exist_ok=True)


def f(x, sig=3):
    """Format a number for LaTeX: plain for 1e-3..1e4, otherwise a x 10^b."""
    x = float(x)
    if x == 0:
        return "0"
    if not np.isfinite(x):
        return r"\infty"
    a = abs(x)
    if 1e-3 <= a < 1e4:
        d = max(sig - 1 - int(np.floor(np.log10(a))), 0)
        return f"{x:.{d}f}"
    e = int(np.floor(np.log10(a)))
    m = x / 10 ** e
    return f"{m:.{sig - 1}f}\\times10^{{{e}}}"


def pct(x, d=1):
    return f"{100 * float(x):.{d}f}"


macros = {}


def mac(name, val):
    assert name.isalpha(), name
    macros[name] = val


def main():
    inf = io.info()
    gamj = io.load("gamma.json")
    V = io.load("verification.json")
    P = io.load("primary.json")
    MC = io.load("multicase.json")
    CP = io.load("control_primary.json")
    CA = io.load("control_alltimes.json")["aggregate"]
    VA = io.load("variants.json")

    # ---- data
    mac("NTimes", str(inf["n_times"]))
    mac("NMembers", str(inf["n_members"]))
    mac("GridSpacing", f(inf["grid_spacing_deg"], 2))
    mac("LowTime", inf["low_time"].replace("T", " ")[:16] + " UTC")
    mac("HighTime", inf["high_time"].replace("T", " ")[:16] + " UTC")
    mac("LowSpread", f(inf["low_spread"], 3))
    mac("HighSpread", f(inf["high_spread"], 3))
    mac("MinSpread", f(inf["min_spread"], 3))
    mac("MaxSpread", f(inf["max_spread"], 3))
    mac("MedianSpread", f(inf["median_spread"], 3))
    n_mc = MC["all"]["n"]
    mac("NCases", f"{n_mc:,}")
    mac("NGradients", f"{n_mc * inf['n_members']:,}")
    mac("NLocs", str(len(inf["locations"])))

    # ---- regularization
    mac("GammaT", f(gamj["gamma"][0], 4))
    mac("GammaTau", f(gamj["gamma"][1], 4))
    mac("BalRatio", f"{M.RHO_BAL:.2f}")

    # ---- verification
    rows = V["rows"]
    orders = np.array([r["order"] for r in rows])
    relmax = np.array([r["rel_max"] for r in rows])
    mac("TaylorOrderMin", f"{orders.min():.3f}")
    mac("TaylorOrderMax", f"{orders.max():.3f}")
    mac("TaylorOrderRange", f"{orders.min():.3f}" if f"{orders.min():.3f}" == f"{orders.max():.3f}"
        else f"{orders.min():.3f}" + "\\text{--}" + f"{orders.max():.3f}")
    mac("FDRelMax", f(relmax.max(), 2))
    for case in ("LOW", "HIGH"):
        ca = V["continuous_adjoint"][case]
        key = "Low" if case == "LOW" else "High"
        mac(f"Quad{key}One", f(ca["1001"]["max_rel_vs_discrete"], 2))
        mac(f"Quad{key}Four", f(ca["4001"]["max_rel_vs_discrete"], 2))
        mac(f"DtErr{key}", f(V["dt_refinement"][case]["max_rel_err"][1], 2))  # 1000 vs 8000 steps

    lines = [r"\begin{tabular}{@{}llcccc@{}}", r"\toprule",
             r"Case & $k$ & Taylor order & \multicolumn{3}{c}{$|a_i-f_i|/|f_i|$ (scaled coordinates)}\\",
             r"\cmidrule(l){4-6}", r" & & & $i=T$ & $\max_{i\in\tau}$ & $\max_i$\\", r"\midrule"]
    for r in rows:
        rel = np.asarray(r["rel"])
        lines.append(f"{r['case']} & ${r['member']}$ & ${r['order']:.3f}$ & ${f(rel[0], 2)}$ & "
                     f"${f(rel[1:].max(), 2)}$ & ${f(rel.max(), 2)}$\\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "tab_verification.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # ---- primary cases
    for case in ("LOW", "HIGH"):
        p = P[case]
        k = "Low" if case == "LOW" else "High"
        b = p["bootstrap"]
        mac(f"{k}Jmean", f(p["J_mean"], 4))
        mac(f"{k}Jstd", f(p["J_std"], 3))
        mac(f"{k}PosDisp", f(p["pos_disp"], 3))
        mac(f"{k}ErrMean", f(p["err_mean"], 3))
        mac(f"{k}Fro", f(p["fro"], 3))
        mac(f"{k}LamOne", f(p["lam1"], 3))
        mac(f"{k}LamTwo", f(p["lam2"], 3))
        mac(f"{k}LamHidden", f(p["lam_hidden"], 3))
        mac(f"{k}RhoHidden", f(p["rho_hidden"], 3))
        mac(f"{k}KappaHidden", f(p["kappa_hidden"], 3))
        mac(f"{k}HiddenRMS", f(np.sqrt(p["lam_hidden"]), 3))
        mac(f"{k}SigmaMeanDir", f(p["sigma_mean_dir"], 3))
        mac(f"{k}Cnorm", f(p["c_norm"], 4))
        mac(f"{k}TopTwo", pct(p["top2_share"], 4))
        mac(f"{k}PropTwoErr", f(p["prop2_rel_err"], 3))
        mac(f"{k}EpsW", f(p["eps_linfit"], 3))
        mac(f"{k}EpsDrag", f(p["eps_drag"], 3))
        mac(f"{k}RtwoDragMin", f(min(p["r2_drag"]), 3))
        mac(f"{k}RtwoDragMax", f(max(p["r2_drag"]), 3))
        mac(f"{k}WbarSpeed", f(p["wbar_speed"], 3))
        mac(f"{k}SpreadRatio", f(p["spread_ratio"], 3))
        mac(f"{k}DMinusOneJstd", f(p["J_std"] * np.sqrt(49 / 50), 4))
        mac(f"{k}BetaStar", f(p["beta_star"], 3))
        mac(f"{k}LooMax", pct(p["loo"]["max_abs"], 1))
        mac(f"{k}LooArg", str(p["loo"]["argmax"]))
        mac(f"{k}AngleKw", f"{p['angle_v1_vs_K_wind_axis']:.1f}")
        mac(f"{k}AngleCI", f"{b['angle_v1']['hi']:.1f}")
        mac(f"{k}ScaleAngleMed", f"{p['scaling']['angle_med']:.1f}")
        mac(f"{k}ScaleAngleP", f"{p['scaling']['angle_p95']:.1f}")
        mac(f"{k}SlopeMean", f"{p['alpha_sweep']['box']['slope_mean']:.2f}")
        mac(f"{k}SlopeStd", f"{p['alpha_sweep']['box']['slope_std']:.2f}")
        sub = {r["n"]: r for r in p["subsets"]}
        mac(f"{k}SubTen", pct(sub[10]["fro_med"], 1))
        mac(f"{k}SubForty", pct(sub[40]["fro_med"], 1))
        mac(f"{k}SubFortyP", pct(sub[40]["fro_p95"], 1))
        mac(f"{k}SubAngleForty", f"{sub[40]['angle_med']:.1f}")
        r0 = p["alpha_sweep"]["box"]["rows"][0]
        mac(f"{k}PredMean", f(r0["pred_mean"], 3))
        mac(f"{k}ActMean", f(r0["act_mean"], 3))
        mac(f"{k}ThrustGrad", f(p["gbar"][0], 3))
        mac(f"{k}HiddenRespMin", f(p["hidden_resp"]["min"], 3))
        mac(f"{k}HiddenRespMax", f(p["hidden_resp"]["max"], 3))
        v1 = np.asarray(p["v1_phys"])
        mac(f"{k}VOne", "(" + ", ".join(f"{x:+.3f}" for x in v1) + ")")
    R = P["ratios"]
    mac("RatioFro", f(R["fro"], 3))
    mac("RatioLamOne", f(R["lam1"], 3))
    mac("RatioHiddenRMS", f(R["hidden_rms"], 3))
    mac("RatioSpread", f(R["spread"], 3))
    mac("RatioSpreadSq", f(R["spread"] ** 2, 3))
    mac("RatioDetSw", f(R["detSw"], 3))
    mac("RatioJstd", f(R["J_std"], 3))

    def ci(p, key, sig=3):
        b = p["bootstrap"][key]
        return f"$[{f(b['lo'], sig)},\\,{f(b['hi'], sig)}]$"

    L, H = P["LOW"], P["HIGH"]
    rows_t = [
        (r"Spread $s$ ($\mathrm{m\,s^{-1}}$)", f"${f(L['spread'])}$", "", f"${f(H['spread'])}$", ""),
        (r"$\det\Sigma_w$ ($\mathrm{m^4\,s^{-4}}$)", f"${f(L['det_Sigma_w'])}$", "", f"${f(H['det_Sigma_w'])}$", ""),
        (r"$\|c\|_2=\|S\bar g\|_2$", f"${f(L['c_norm'], 4)}$", "", f"${f(H['c_norm'], 4)}$", ""),
        (r"$\|H\|_F$", f"${f(L['fro'])}$", ci(L, "fro"), f"${f(H['fro'])}$", ci(H, "fro")),
        (r"$\lambda_1(H)$", f"${f(L['lam1'])}$", ci(L, "lam1"), f"${f(H['lam1'])}$", ci(H, "lam1")),
        (r"$\lambda_2(H)$", f"${f(L['lam2'])}$", "", f"${f(H['lam2'])}$", ""),
        (r"$(\lambda_1+\lambda_2)/\mathrm{tr}\,H$ (\%)", f"${pct(L['top2_share'], 4)}$", "",
         f"${pct(H['top2_share'], 4)}$", ""),
        (r"$\lambda_{\mathrm{hid}}$", f"${f(L['lam_hidden'])}$", ci(L, "lam_hidden"),
         f"${f(H['lam_hidden'])}$", ci(H, "lam_hidden")),
        (r"$\rho_{\mathrm{hid}}=\lambda_{\mathrm{hid}}/\lambda_1$", f"${f(L['rho_hidden'])}$", ci(L, "rho_hidden"),
         f"${f(H['rho_hidden'])}$", ci(H, "rho_hidden")),
        (r"$\kappa_{\mathrm{hid}}=\sqrt{\lambda_{\mathrm{hid}}}/\|c\|_2$", f"${f(L['kappa_hidden'])}$",
         ci(L, "kappa_hidden"), f"${f(H['kappa_hidden'])}$", ci(H, "kappa_hidden")),
        (r"Angle of $v_1$, $97.5\%$ bootstrap ($^\circ$)", f"${L['bootstrap']['angle_v1']['hi']:.1f}$", "",
         f"${H['bootstrap']['angle_v1']['hi']:.1f}$", ""),
        (r"Relative spread $s/\|\bar w\|$", f"${f(L['spread_ratio'])}$", "", f"${f(H['spread_ratio'])}$", ""),
        (r"Tangent error $\epsilon_{\mathrm{lin}}$ (no fit)", f"${f(L['prop2_rel_err'])}$", "",
         f"${f(H['prop2_rel_err'])}$", ""),
        (r"Linear-in-$w$ error $\epsilon_{w}$", f"${f(L['eps_linfit'])}$", "", f"${f(H['eps_linfit'])}$", ""),
        (r"Drag-feature error $\epsilon_{\mathrm{drag}}$", f"${f(L['eps_drag'])}$", "", f"${f(H['eps_drag'])}$", ""),
        (r"Max. leave-one-out change of $\|H\|_F$ (\%)",
         f"${pct(L['loo']['max_abs'])}$ ($k={L['loo']['argmax']}$)", "",
         f"${pct(H['loo']['max_abs'])}$ ($k={H['loo']['argmax']}$)", ""),
        (r"Threshold $\beta^*$ (Prop.~3)", f"${f(L['beta_star'])}$", "", f"${f(H['beta_star'])}$", ""),
    ]
    lines = [r"\begin{tabular}{@{}lcccc@{}}", r"\toprule",
             r"Quantity & LOW & $95\%$ CI & HIGH & $95\%$ CI\\", r"\midrule"]
    lines += [" & ".join(r) + r"\\" for r in rows_t]
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "tab_primary.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # ---- multicase
    for scope, k in (("site", "Site"), ("all", "All")):
        m = MC[scope]
        mac(f"MC{k}N", str(m["n"]))
        mac(f"MC{k}RhoLamTr", f"{m['spearman_lam1_trSw']:.2f}")
        mac(f"MC{k}RhoLamDet", f"{m['spearman_lam1_detSw']:.2f}")
        mac(f"MC{k}RhoLamSpread", f"{m['spearman_lam1_spread']:.2f}")
        mac(f"MC{k}RhoHidSpread", f"{m['spearman_hidden_spread']:.2f}")
        mac(f"MC{k}RhoJstdLam", f"{m['spearman_Jstd_lam1']:.2f}")
        mac(f"MC{k}PropMed", f(m["prop2_err_median"], 3))
        mac(f"MC{k}PropP", f(m["prop2_err_p95"], 3))
        mac(f"MC{k}PropMax", f(m["prop2_err_max"], 3))
        mac(f"MC{k}TopTwoMin", pct(m["top2_share_min"], 3))
        mac(f"MC{k}RhoHidMed", f(m["rho_hidden_median"], 3))
        mac(f"MC{k}RhoHidMin", f(m["rho_hidden_range"][0], 3))
        mac(f"MC{k}KappaMed", f(m["kappa_hidden_median"], 3))
        mac(f"MC{k}KappaP", f(m["kappa_hidden_p95"], 3))
        mac(f"MC{k}EpsWMed", f(m["eps_linfit_median"], 3))
        mac(f"MC{k}EpsDragMed", f(m["eps_drag_median"], 3))
        mac(f"MC{k}EpsDragP", f(m["eps_drag_p95"], 3))
        mac(f"MC{k}RtwoDragMed", f(m["loo_r2_drag_median"], 3))
        mac(f"MC{k}RhoPropRatio", f"{m['spearman_prop2_ratio']:.2f}")
        mac(f"MC{k}FracRatioHalf", pct(m["frac_ratio_gt_half"], 0))
        mac(f"MC{k}PropMedSmallRatio", f(m["prop2_err_median_ratio_lt_02"], 3))
        mac(f"MC{k}NSmallRatio", str(m["n_ratio_lt_02"]))
    mac("MCSlope", f"{MC['loglog_lam1_trSw_slope']:.2f}")

    # ---- control
    names = [("mean-box", r"Mean, box ($\beta=0$)"), ("mean-dir", r"Mean, direction"),
             ("CASS-1 (b=1.645)", r"CASS-1, $\beta=1.645$"), ("CASS-1 (b=2b*)", r"CASS-1, $\beta=2\beta^*$"),
             ("CASS-1 (b=5b*)", r"CASS-1, $\beta=5\beta^*$"), ("mean-2", r"Mean, 2nd order"),
             ("CASS-2 (b=1.645)", r"CASS-2, $\beta=1.645$"), ("SAA", r"SAA (nonlinear)"),
             ("CVaR-0.9", r"$\mathrm{CVaR}_{0.9}$, 2nd order")]
    lines = [r"\begin{tabular}{@{}lrrrrrrr@{}}", r"\toprule",
             r" & \multicolumn{3}{c}{LOW ($100$ splits)} & \multicolumn{3}{c}{HIGH ($100$ splits)} & All $62$\\",
             r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(l){8-8}",
             r"Rule & $\Delta\bar J$ & $\Delta J_{95}$ & $\|x\|_2$ & $\Delta\bar J$ & $\Delta J_{95}$ & $\|x\|_2$ & $\Delta J_{95}$\\",
             r"\midrule"]
    for key, lab in names:
        if key not in CP["LOW"]["summary"]:
            continue
        l, h = CP["LOW"]["summary"][key], CP["HIGH"]["summary"][key]
        a = CA.get(key, {}).get("d_p95", np.nan)
        lines.append(f"{lab} & ${l['d_mean'] * 1e3:+.1f}$ & ${l['d_p95'] * 1e3:+.1f}$ & ${l['xnorm2']:.2f}$ & "
                     f"${h['d_mean'] * 1e3:+.1f}$ & ${h['d_p95'] * 1e3:+.1f}$ & ${h['xnorm2']:.2f}$ & "
                     f"${a * 1e3:+.1f}$\\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "tab_control.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    for case in ("LOW", "HIGH"):
        k = "Low" if case == "LOW" else "High"
        s = CP[case]["summary"]
        for key, nm in (("mean-box", "MeanBox"), ("mean-dir", "MeanDir"), ("mean-2", "MeanTwo"),
                        ("CASS-2 (b=1.645)", "CassTwo"), ("SAA", "SAA"), ("CVaR-0.9", "CVaR"),
                        ("CASS-1 (b=1.645)", "CassOne"), ("CASS-1 (b=5b*)", "CassOneFive")):
            if key in s:
                mac(f"{k}{nm}dMean", f"{s[key]['d_mean'] * 1e3:+.1f}")
                mac(f"{k}{nm}dP", f"{s[key]['d_p95'] * 1e3:+.1f}")
                mac(f"{k}{nm}Xn", f"{s[key]['xnorm2']:.2f}")
                mac(f"{k}{nm}Exceed", pct(s[key]["exceed"], 0))
        mac(f"{k}NomExceed", pct(s["nominal"]["exceed"], 0))
        mac(f"{k}BetaStarMed", f(CP[case]["beta_star_median"], 3))
        hd = np.asarray(CP[case]["scaled_mean_hessian_diag"])
        mac(f"{k}QTT", f(hd[0], 3))
    for key, nm in (("mean-box", "MeanBox"), ("mean-2", "MeanTwo"), ("CASS-2 (b=1.645)", "CassTwo"),
                    ("SAA", "SAA"), ("CVaR-0.9", "CVaR"), ("CASS-1 (b=1.645)", "CassOne"),
                    ("CASS-1 (b=5b*)", "CassOneFive"), ("mean-dir", "MeanDir")):
        if key in CA:
            mac(f"All{nm}dP", f"{CA[key]['d_p95'] * 1e3:+.1f}")
            mac(f"All{nm}dMean", f"{CA[key]['d_mean'] * 1e3:+.1f}")
            mac(f"All{nm}Win", pct(CA[key]["frac_cases_improve_p95"], 0))

    # ---- paired comparisons over all site times
    PT = io.load("control_alltimes.json")["per_time"]

    def paired(a, b):
        return np.array([t["summary"][a]["d_p95"] - t["summary"][b]["d_p95"] for t in PT
                         if a in t["summary"] and b in t["summary"]])
    d1 = paired("CASS-1 (b=1.645)", "mean-box")
    mac("AllNBetaBelow", str(int(np.sum(np.array([t["beta_star_median"] for t in PT]) < 1.645))))
    tol = 1e-6                     # differences below this are SOCP solver tolerance (identical steps)
    mac("AllCassOneBetter", str(int(np.sum(d1 < -tol))))
    mac("AllCassOneWorse", str(int(np.sum(d1 > tol))))
    mac("AllCassOneGainMax", f"{-d1.min() * 1e3:.1f}")
    mac("AllCassOneLossMax", f"{d1.max() * 1e3:.1f}")
    mac("AllCassOneGainMean", f"{-d1.mean() * 1e3:.2f}")
    d12 = paired("CASS-1 (b=1.645)", "mean-2")
    mac("AllCassOneVsTwoMin", f"{d12.min() * 1e3:.1f}")
    d22 = paired("CASS-2 (b=1.645)", "mean-2")
    mac("AllCassTwoVsTwoMax", f(d22.max(), 2))
    dbm = paired("mean-2", "mean-box")
    mac("AllCurvGainMean", f"{-dbm.mean() * 1e3:.1f}")
    mac("AllBetaStarMin", f(min(t["beta_star_median"] for t in PT), 2))
    mac("AllBetaStarMax", f(max(t["beta_star_median"] for t in PT), 3))

    # ---- variants
    order = ["open-loop", "rotor drag", "closed loop", "near-optimum"]
    lines = [r"\begin{tabular}{@{}lcccccc@{}}", r"\toprule",
             r"Variant & \multicolumn{2}{c}{$\lambda_1(H)$} & HIGH/LOW & \multicolumn{2}{c}{$\epsilon_{\mathrm{lin}}$} & $\frac{\lambda_1+\lambda_2}{\mathrm{tr}\,H}$ (\%)\\",
             r"\cmidrule(lr){2-3}\cmidrule(lr){5-6}", r" & LOW & HIGH & & LOW & HIGH & min\\", r"\midrule"]
    for v in order:
        d = VA[v]
        lines.append(f"{v.capitalize()} & ${f(d['LOW']['lam1'])}$ & ${f(d['HIGH']['lam1'])}$ & "
                     f"${f(d['ratio_lam1'])}$ & ${f(d['LOW']['prop2_rel_err'])}$ & "
                     f"${f(d['HIGH']['prop2_rel_err'])}$ & "
                     f"${pct(min(d['LOW']['top2_share'], d['HIGH']['top2_share']), 3)}$\\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "tab_variants.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    for v, nm in (("rotor drag", "Rotor"), ("closed loop", "Closed"), ("near-optimum", "Opt")):
        mac(f"Var{nm}Ratio", f(VA[v]["ratio_lam1"], 3))
        mac(f"Var{nm}ErrMax", f(max(VA[v]["LOW"]["prop2_rel_err"], VA[v]["HIGH"]["prop2_rel_err"]), 3))
    no = VA["near-optimum"]
    mac("OptCnormLow", f(no["LOW"]["c_norm"], 2))
    mac("OptCnormHigh", f(no["HIGH"]["c_norm"], 2))
    mac("OptJLow", f(no["LOW"]["J_mean"], 4))
    mac("OptJHigh", f(no["HIGH"]["J_mean"], 4))

    # ---- supplementary raw-coordinate table
    lines = [r"\begin{tabular}{@{}lcc@{}}", r"\toprule", r"Quantity (raw units) & LOW & HIGH\\", r"\midrule"]
    for i, nm in enumerate(["T", r"\tau_x", r"\tau_y", r"\tau_z"]):
        lines.append(f"$\\bar g_{{{nm}}}$ & ${f(L['gbar'][i], 5)}$ & ${f(H['gbar'][i], 5)}$\\\\")
    lines.append(f"$\\|D\\|_F$ (mixed units), divisor $E-1$ & ${f(L['raw_fro'], 4)}$ & ${f(H['raw_fro'], 4)}$\\\\")
    lines.append(f"$\\|D\\|_F$ (mixed units), divisor $E$ & ${f(L['raw_fro'] * 49 / 50, 4)}$ & ${f(H['raw_fro'] * 49 / 50, 4)}$\\\\")
    lines.append(f"$\\|H\\|_F$, divisor $E$ & ${f(L['fro_ddof0'], 4)}$ & ${f(H['fro_ddof0'], 4)}$\\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "tab_supp_raw.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # ---- appendix: coordinates of all locations of the multi-case analysis
    z = io.load_winds()
    nla, nlo = np.asarray(z["nbhd_lat"], float), np.asarray(z["nbhd_lon"], float)
    clat, clon = float(nla[len(nla) // 2]), float(nlo[len(nlo) // 2])

    def deg(v, ns):
        hemi = ns[0] if v >= 0 else ns[1]
        return f"${abs(v):.1f}^\\circ$\\,{hemi}"
    lines = [r"\begin{tabular}{@{}lccc@{}}", r"\toprule",
             r"Location & Latitude & Longitude & Grid points\\", r"\midrule",
             f"Study site (center) & {deg(clat, 'NS')} & {deg(clon, 'EW')} & (in the $25$)\\\\",
             f"Site neighborhood ($5\\times5$) & "
             f"${nla.min():.1f}$--${nla.max():.1f}^\\circ$\\,N & ${nlo.min():.1f}$--${nlo.max():.1f}^\\circ$\\,E & ${nla.size * nlo.size}$\\\\"]
    for name, (la, lo) in zip(z["loc_names"], np.asarray(z["loc_latlon"], float)):
        lines.append(f"{name} & {deg(la, 'NS')} & {deg(lo, 'EW')} & $1$\\\\")
    lines += [r"\midrule", f"Total & & & ${nla.size * nlo.size + len(z['loc_names'])}$\\\\",
              r"\bottomrule", r"\end{tabular}"]
    (OUT / "tab_locations.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    with open(OUT / "numbers.tex", "w", encoding="utf-8") as fh:
        fh.write("% generated by scripts/s07_tables.py -- do not edit by hand\n")
        for k, v in macros.items():
            fh.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
    print(len(macros), "macros written")


if __name__ == "__main__":
    main()
