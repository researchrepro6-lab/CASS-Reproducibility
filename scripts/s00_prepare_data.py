"""Step 0 - build processed wind arrays from the extracted ECMWF subset.

Input : data/raw/cass_wind_subset.npz   (made by extract_cass_subset.py from
        ECMWF_Aligned_July2022.nc; TIGGE ECMWF ENS, 50 perturbed members,
        step = 0, July 2022, 00/12 UTC)
Output: data/processed/winds.npz with
        times (T,), members (E,)
        site  (T, E, 2)            u10, v10 at 28.0N 95.5E
        nbhd  (T, 5, 5, E, 2)      5 x 5 full-resolution neighbourhood
        locs  (L, T, E, 2)         contrasting locations on the coarse grid
        loc_latlon (L, 2)
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cass  # noqa: E402,F401  (checks that the required packages are installed)
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/cass_wind_subset.npz"
OUT = ROOT / "data/processed/winds.npz"
LAT0, LON0 = 28.0, 95.5
# contrasting regimes inside the 0-30N, 60-120E domain (nearest coarse-grid node)
TARGET_LOCS = [
    ("Arabian Sea", 15.0, 65.0),
    ("Equatorial Indian Ocean", 2.5, 75.0),
    ("Thar Desert", 27.5, 70.0),
    ("Bay of Bengal", 15.0, 87.5),
    ("Tibetan Plateau", 30.0, 90.0),
    ("Indochina", 15.0, 105.0),
    ("South China Sea", 10.0, 115.0),
]


def main():
    z = np.load(RAW, allow_pickle=True)
    order = list(z["dim_order"])
    assert len(order) == 4, f"expected member,time,lat,lon got {order}"
    su, sv = z["site_u10"], z["site_v10"]          # (E, T, nlat, nlon)
    slat, slon = z["site_lat"], z["site_lon"]
    i0 = int(np.argmin(abs(slat - LAT0)))
    j0 = int(np.argmin(abs(slon - LON0)))
    assert abs(slat[i0] - LAT0) < 1e-6 and abs(slon[j0] - LON0) < 1e-6, "site not on grid"

    def tme(u, v):  # (E,T,...) -> (T,...,E,2)
        a = np.stack([u, v], -1)
        return np.moveaxis(a, 0, -2)

    site = tme(su[:, :, i0, j0], sv[:, :, i0, j0])                       # (T,E,2)
    nb = tme(su[:, :, i0 - 2:i0 + 3, j0 - 2:j0 + 3], sv[:, :, i0 - 2:i0 + 3, j0 - 2:j0 + 3])  # (T,5,5,E,2)

    cu, cv = z["coarse_u10"], z["coarse_v10"]
    clat, clon = z["coarse_lat"], z["coarse_lon"]
    locs, ll, names = [], [], []
    for name, la, lo in TARGET_LOCS:
        i = int(np.argmin(abs(clat - la)))
        j = int(np.argmin(abs(clon - lo)))
        locs.append(tme(cu[:, :, i, j], cv[:, :, i, j]))
        ll.append((float(clat[i]), float(clon[j])))
        names.append(name)
    locs = np.asarray(locs)

    times = np.asarray(z["times"]).astype(str)
    members = np.asarray(z["members"])
    assert np.isfinite(site).all() and np.isfinite(nb).all() and np.isfinite(locs).all()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT, times=times, members=members, site=site, nbhd=nb,
                        nbhd_lat=slat[i0 - 2:i0 + 3], nbhd_lon=slon[j0 - 2:j0 + 3],
                        locs=locs, loc_latlon=np.asarray(ll), loc_names=np.asarray(names),
                        grid_spacing=z["grid_spacing_deg"])

    # spread ranking (paper Section V-A)
    spread = np.array([np.mean(np.linalg.norm(site[t] - site[t].mean(0), axis=1)) for t in range(len(times))])
    rank = np.argsort(spread)
    info = dict(n_times=len(times), n_members=int(site.shape[1]),
                grid_spacing_deg=float(z["grid_spacing_deg"]),
                low_time=times[rank[1]], low_spread=float(spread[rank[1]]),
                high_time=times[rank[-2]], high_spread=float(spread[rank[-2]]),
                min_spread=float(spread.min()), max_spread=float(spread.max()),
                median_spread=float(np.median(spread)),
                locations=[dict(name=n, lat=a, lon=b) for n, (a, b) in zip(names, ll)])
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results/data_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    np.save(ROOT / "results/site_spread.npy", spread)
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    sys.exit(main())
