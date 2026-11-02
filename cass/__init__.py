"""CASS library. Importing it first checks that the required packages are installed
for the Python that runs the script, and explains how to install them if not."""
import importlib.util as _util
import sys as _sys
from pathlib import Path as _Path

_REQUIRED = ("numpy", "scipy", "jax", "jaxlib", "cvxpy", "clarabel", "matplotlib")
_missing = [m for m in _REQUIRED if _util.find_spec(m) is None]
if _missing:
    _exe = _Path(_sys.executable)
    if _exe.name.lower() == "pythonw.exe" and _exe.with_name("python.exe").exists():
        _exe = _exe.with_name("python.exe")
    _req = _Path(__file__).resolve().parents[1] / "requirements.txt"
    raise SystemExit(
        f"\nThe package(s) {', '.join(_missing)} are not installed for this Python:\n    {_exe}\n\n"
        "Install them once, in either way:\n"
        "  * open install_requirements.py (in the package folder) in IDLE and press F5, or\n"
        f'  * in a Command Prompt run:  "{_exe}" -m pip install -r "{_req}"\n')
