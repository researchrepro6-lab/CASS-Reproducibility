#!/usr/bin/env bash
# Linux/macOS launcher; all logic is in run_all.py (works on Windows too: python run_all.py).
#   bash run_all.sh --check | --quick | (no option: full run)
cd "$(dirname "$0")"
exec python3 run_all.py "$@"
