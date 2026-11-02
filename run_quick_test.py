"""Quick smoke test of every script (about 5-10 min).

In IDLE: open this file and press F5.  In a terminal: python run_quick_test.py
The test runs in a temporary copy, so results/, figures/ and tables/ are not changed.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_all  # noqa: E402

if __name__ == "__main__":
    sys.argv = [sys.argv[0], "--quick"]
    try:
        run_all.main()
    except SystemExit as exc:              # keep IDLE quiet about the normal exit
        if exc.code not in (0, None):
            print(f"\nQuick test ended with exit code {exc.code}.")
