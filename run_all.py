"""Cross-platform runner for the CASS reproducibility package (Windows, macOS, Linux).

Usage (from this folder):
    python run_all.py --check      check Python, packages and data; nothing is run
    python run_all.py --quick      smoke test of every script (~5-10 min); runs in a temporary
                                   copy, so the shipped results/, figures/ and tables/ stay untouched
    python run_all.py              full run (~1.5-2 h on 2 CPU cores); regenerates results/,
                                   figures/ and tables/ in place and compares the regenerated
                                   numbers with the ones that were shipped
    python run_all.py --from s06   full run starting at step s06 (uses earlier results on disk)

In IDLE: open this file and press F5 for the full run, or use Run > Run... Customized
(Shift+F5) and type --check or --quick. run_quick_test.py does the quick test with F5.

If the manuscript LaTeX source has been unpacked into paper/, the full run also compiles it
(requires pdflatex and bibtex on the PATH).
"""
import argparse
import importlib
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STEPS = [
    ("s00", "s00_prepare_data.py", "processed winds, spread ranking, LOW/HIGH selection"),
    ("s01", "s01_verify.py", "regularization weights and gradient verification"),
    ("s02", "s02_primary.py", "LOW/HIGH CASS statistics, bootstrap, subsets, scaling"),
    ("s03", "s03_multicase.py", "all 1,984 cases (longest step)"),
    ("s04", "s04_control.py", "out-of-sample decision experiment (long step)"),
    ("s05", "s05_variants.py", "model variants"),
    ("s06", "s06_figures.py", "figures (600-dpi PNG)"),
    ("s07", "s07_tables.py", "numbers and LaTeX tables"),
]
PACKAGES = [("numpy", "numpy"), ("scipy", "scipy"), ("jax", "jax"), ("jaxlib", "jaxlib"),
            ("cvxpy", "cvxpy"), ("clarabel", "clarabel"), ("matplotlib", "matplotlib")]
DATA = ROOT / "data" / "raw" / "cass_wind_subset.npz"


def python_exe():
    """The console interpreter (IDLE on Windows runs code under pythonw.exe)."""
    exe = Path(sys.executable)
    if exe.name.lower() == "pythonw.exe" and exe.with_name("python.exe").exists():
        return str(exe.with_name("python.exe"))
    return str(exe)


def check_environment():
    ok = True
    print(f"Python {sys.version.split()[0]}, {struct.calcsize('P') * 8}-bit ({python_exe()})")
    if sys.version_info < (3, 10) or sys.version_info >= (3, 15):
        print("  ERROR: use Python 3.10-3.14 (JAX/CVXPY do not provide packages for this version).")
        ok = False
    if struct.calcsize("P") * 8 != 64:
        print("  ERROR: 64-bit Python is required (JAX has no 32-bit build).")
        ok = False
    missing = False
    for mod, pipname in PACKAGES:
        try:
            m = importlib.import_module(mod)
            print(f"  {mod:11s} {getattr(m, '__version__', '?')}")
        except Exception as exc:                        # noqa: BLE001
            print(f"  {mod:11s} MISSING ({exc.__class__.__name__})")
            missing = True
    if missing:
        print("\n  Some packages are missing for THIS Python. Install them by running\n"
              "  install_requirements.py (open it in IDLE and press F5), or in a terminal:\n"
              f'    "{python_exe()}" -m pip install -r requirements.txt')
        return False
    if not ok:
        return False
    if ok:
        import cvxpy as cp
        import jax
        jax.config.update("jax_enable_x64", True)
        import jax.numpy as jnp
        if jnp.ones(1).dtype != jnp.float64:
            print("  ERROR: JAX could not enable float64.")
            ok = False
        if "CLARABEL" not in cp.installed_solvers():
            print("  ERROR: CVXPY does not see the Clarabel solver -> pip install clarabel")
            ok = False
    latex = shutil.which("latex")
    print(f"  LaTeX for figures: {'found' if latex else 'not found (optional; figures then use the built-in renderer)'}")
    if DATA.exists():
        print(f"  input data: {DATA.relative_to(ROOT)} ({DATA.stat().st_size / 1e6:.1f} MB)")
    else:
        print(f"  ERROR: missing input data {DATA.relative_to(ROOT)} (see README)")
        ok = False
    return ok


def read_numbers(path):
    if not path.exists():
        return {}
    txt = path.read_text(encoding="utf-8")
    return dict(re.findall(r"^\\newcommand\{\\(\w+)\}\{(.*)\}\s*$", txt, flags=re.M))


def run_steps(base, steps, env):
    for key, script, what in steps:
        print(f"\n=== {key}: {script} - {what}", flush=True)
        t0 = time.time()
        proc = subprocess.Popen([python_exe(), str(base / "scripts" / script)], cwd=base, env=env,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, encoding="utf-8", errors="replace")
        for line in proc.stdout:                    # shown in a terminal and in IDLE alike
            print(line, end="", flush=True)
        if proc.wait() != 0:
            print(f"\nFAILED at {script} (exit code {proc.returncode}). The error message is printed above.")
            return False
        print(f"--- {key} finished in {time.time() - t0:.0f} s", flush=True)
    return True


def compile_paper(base):
    paper = base / "paper"
    if not (paper / "main.tex").exists():
        return
    if not shutil.which("pdflatex") or not shutil.which("bibtex"):
        print("\npaper/ found but pdflatex/bibtex not on PATH; skipping compilation.")
        return
    shutil.rmtree(paper / "figures", ignore_errors=True)
    shutil.copytree(base / "figures", paper / "figures")
    (paper / "generated").mkdir(exist_ok=True)
    for f in (base / "tables").glob("*.tex"):
        shutil.copy2(f, paper / "generated" / f.name)
    for cmd in (["pdflatex", "-interaction=nonstopmode", "main.tex"], ["bibtex", "main"],
                ["pdflatex", "-interaction=nonstopmode", "main.tex"],
                ["pdflatex", "-interaction=nonstopmode", "main.tex"],
                ["pdflatex", "-interaction=nonstopmode", "main.tex"]):   # float labels settle after 4 passes
        subprocess.run(cmd, cwd=paper, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print("\nCompiled paper/main.pdf")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="only check the environment")
    ap.add_argument("--quick", action="store_true", help="smoke test in a temporary copy")
    ap.add_argument("--from", dest="start", default="s00", help="first step to run (s00..s07)")
    args = ap.parse_args()

    print("CASS reproducibility package\n")
    if not check_environment():
        sys.exit(1)
    if args.check:
        print("\nEnvironment OK.")
        return

    keys = [k for k, _, _ in STEPS]
    if args.start not in keys:
        sys.exit(f"--from must be one of {keys}")
    steps = STEPS[keys.index(args.start):]
    env = dict(os.environ, PYTHONIOENCODING="utf-8", MPLBACKEND="Agg")

    if args.quick:
        env["CASS_QUICK"] = "1"
        tmp = Path(tempfile.mkdtemp(prefix="cass_quick_"))
        base = tmp / "bundle"
        shutil.copytree(ROOT, base, ignore=shutil.ignore_patterns("paper", "__pycache__", ".git", "*.pyc"))
        print(f"\nQuick smoke test in a temporary copy: {base}")
        ok = run_steps(base, steps, env)
        shutil.rmtree(tmp, ignore_errors=True)
        print("\nQuick test PASSED: every script ran without error." if ok else "\nQuick test FAILED.")
        sys.exit(0 if ok else 1)

    before = read_numbers(ROOT / "tables" / "numbers.tex")
    t0 = time.time()
    if not run_steps(ROOT, steps, env):
        sys.exit(1)
    after = read_numbers(ROOT / "tables" / "numbers.tex")
    print(f"\nAll steps finished in {(time.time() - t0) / 60:.1f} min.")
    if before and after:
        same = [k for k in after if before.get(k) == after[k]]
        diff = [k for k in after if k in before and before[k] != after[k]]
        print(f"Regenerated numbers identical to the shipped ones: {len(same)} of {len(after)}")
        for k in diff[:20]:
            print(f"  differs: {k}: shipped {before[k]}  ->  now {after[k]}")
        if diff:
            print("  (differences in the last digit can come from other library versions or CPUs)")
    compile_paper(ROOT)


if __name__ == "__main__":
    main()
