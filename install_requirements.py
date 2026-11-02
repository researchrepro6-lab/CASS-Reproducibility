"""Install the required packages for THIS Python (the one that runs this file).

In IDLE:      open this file and press F5.
In a terminal: python install_requirements.py

Afterwards it runs the environment check of run_all.py.
"""
import struct
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def python_exe():
    exe = Path(sys.executable)
    if exe.name.lower() == "pythonw.exe" and exe.with_name("python.exe").exists():
        return str(exe.with_name("python.exe"))       # IDLE on Windows runs under pythonw.exe
    return str(exe)


def run(cmd):
    print("\n> " + " ".join(f'"{c}"' if " " in c else c for c in cmd), flush=True)
    proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, encoding="utf-8", errors="replace")
    out = []
    for line in proc.stdout:
        print(line, end="", flush=True)
        out.append(line)
    return proc.wait(), "".join(out)


def main():
    v, bits = sys.version_info, struct.calcsize("P") * 8
    print(f"Installing the CASS requirements for Python {v.major}.{v.minor}.{v.micro} ({bits}-bit):\n  {python_exe()}")
    if v < (3, 10) or v >= (3, 15):
        print("\nThis Python version is not supported (JAX/CVXPY provide packages for Python 3.10-3.14).\n"
              "Install Python 3.12 or 3.13 (64-bit) from https://www.python.org and run this file with it.")
        return
    if bits != 64:
        print("\n32-bit Python is not supported by JAX. Install 64-bit Python 3.12 or 3.13 from https://www.python.org.")
        return
    run([python_exe(), "-m", "pip", "install", "--upgrade", "pip"])
    code, out = run([python_exe(), "-m", "pip", "install", "-r", str(ROOT / "requirements.txt")])
    if code != 0 and ("Permission" in out or "Access is denied" in out or "WinError 5" in out):
        print("\nNo permission to install for all users; installing for the current user only.")
        code, out = run([python_exe(), "-m", "pip", "install", "--user", "-r", str(ROOT / "requirements.txt")])
    if code != 0:
        print("\nInstallation FAILED (see the messages above). Common causes: no internet connection, "
              "a proxy/firewall, or an unsupported Python version.")
        return
    print("\nPackages installed. Checking the environment ...")
    run([python_exe(), str(ROOT / "run_all.py"), "--check"])
    print("\nNext: open run_quick_test.py and press F5 (about 5-10 min), or run_all.py for the full run.")


if __name__ == "__main__":
    main()
