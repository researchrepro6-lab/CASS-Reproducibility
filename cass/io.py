"""Shared I/O helpers."""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
FIG = ROOT / "figures"


def load_winds():
    return np.load(ROOT / "data/processed/winds.npz", allow_pickle=True)


def info():
    return json.loads((RES / "data_info.json").read_text(encoding="utf-8"))


def primary_cases():
    """Return {'LOW': W (E,2), 'HIGH': W} and their timestamps."""
    z = load_winds()
    inf = info()
    times = list(z["times"])
    out = {}
    for case, key in (("LOW", "low_time"), ("HIGH", "high_time")):
        t = times.index(inf[key])
        out[case] = dict(W=np.asarray(z["site"][t], float), time=inf[key], t_index=t)
    return out


def gamma():
    return np.asarray(json.loads((RES / "gamma.json").read_text(encoding="utf-8"))["gamma"])


class NpEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)


def dump(obj, name):
    RES.mkdir(exist_ok=True)
    (RES / name).write_text(json.dumps(obj, indent=2, cls=NpEncoder), encoding="utf-8")


def load(name):
    return json.loads((RES / name).read_text(encoding="utf-8"))
