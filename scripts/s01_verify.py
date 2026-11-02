"""Step 1 - regularization weights and gradient verification.

* gamma from the gradient-balancing rule (LOW case, u0)
* Taylor remainder test (order 2 expected) for members 1, 13, 25, 37, 50 of LOW and HIGH
* component-wise central differences in scaled coordinates (same RK4 solver)
* RK4 step refinement
* independent continuous adjoint (DOP853, AD Jacobians) with quadrature refinement
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cass  # noqa: E402,F401  (checks that the required packages are installed)
import numpy as np

from cass import io, model as M, verify as V

MEMBERS = [1, 13, 25, 37, 50]


def main():
    cases = io.primary_cases()
    gam, gt = M.balance_gamma(cases["LOW"]["W"])
    io.dump(dict(gamma=gam, rule=f"||grad R|| = {M.RHO_BAL} ||mean tracking grad|| per block at u0 (LOW)",
                 mean_tracking_grad_low=gt), "gamma.json")
    print("gamma", gam)

    rows, taylor = [], {}
    for case, d in cases.items():
        for k in MEMBERS:
            w = d["W"][k - 1]
            tt = V.taylor_test(w, gam, seed=k)
            fd = V.fd_componentwise(w, gam)
            rows.append(dict(case=case, member=k, order=tt["order"], rel=fd["rel"],
                             rel_max=float(fd["rel"].max()), rel_norm=fd["rel_norm"],
                             adj=fd["adj"], fd=fd["fd"]))
            taylor[f"{case}_{k}"] = dict(h=tt["h"], r0=tt["r0"], r1=tt["r1"], order=tt["order"])
            print(case, k, "order %.3f" % tt["order"], "max comp rel err %.2e" % fd["rel"].max())

    dtref = {case: V.dt_refinement(d["W"][0], gam) for case, d in cases.items()}
    cont = {}
    for case, d in cases.items():
        w = d["W"][0]
        ca = V.continuous_adjoint(w, gam)
        g_fine = dtref[case]["grads"][-1]
        cont[case] = {str(n): dict(g=g, max_rel_vs_discrete=float(np.max(np.abs(g - g_fine) / np.abs(g_fine))))
                      for n, g in ca.items()}
        print(case, {n: v["max_rel_vs_discrete"] for n, v in cont[case].items()})

    io.dump(dict(rows=rows, taylor=taylor,
                 dt_refinement={c: dict(steps=v["steps"], max_rel_err=v["max_rel_err"]) for c, v in dtref.items()},
                 continuous_adjoint=cont), "verification.json")


if __name__ == "__main__":
    main()
