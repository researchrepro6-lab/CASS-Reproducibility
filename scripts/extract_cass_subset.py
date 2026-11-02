"""
extract_cass_subset.py
----------------------
Cuts ECMWF_Aligned_July2022.nc (~458 MB) down to the small subset needed
for the CASS study (data/raw/cass_wind_subset.npz). Only needed to rebuild the
input from the original archive file; the subset itself is shipped with the package.

Usage:
    pip install xarray netCDF4
    python scripts/extract_cass_subset.py path/to/ECMWF_Aligned_July2022.nc [output_folder]

Output (default folder: data/raw of this package):
    cass_wind_subset.npz      -> u10/v10 for all members and all times on
                                 (a) the full-resolution 9x9 neighbourhood of
                                     the study site (28.0N, 95.5E), and
                                 (b) a coarse 2.5-degree grid over the whole
                                     domain (for multi-location analysis)
    cass_dataset_summary.txt  -> printout of the dataset structure
"""
import sys
from pathlib import Path

import numpy as np

try:
    import xarray as xr
except ImportError:
    print("extract_cass_subset.py is NOT needed to reproduce the results (its output is included).\n"
          "To rebuild the input from the original archive file it needs:  pip install xarray netCDF4")
    sys.exit(0)

SRC = Path(sys.argv[1] if len(sys.argv) > 1 else "ECMWF_Aligned_July2022.nc")
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(__file__).resolve().parents[1] / "data" / "raw"
if not SRC.exists():
    print("extract_cass_subset.py is NOT needed to reproduce the results: its output,\n"
          "data/raw/cass_wind_subset.npz, is already included in this package.\n\n"
          "It only rebuilds that file from the original 458 MB archive file, which was not found:\n"
          f"    {SRC.resolve()}\n"
          "To use it, pass the path of the archive file, e.g.\n"
          "    python scripts/extract_cass_subset.py C:\\data\\ECMWF_Aligned_July2022.nc")
    sys.exit(0)
OUT.mkdir(parents=True, exist_ok=True)
LAT0, LON0 = 28.0, 95.5
HALF = 4          # 9x9 full-resolution box around the site
COARSE_DEG = 2.5  # spacing of the domain-wide coarse grid

ds = xr.open_dataset(SRC)
with open(OUT / "cass_dataset_summary.txt", "w", encoding="utf-8") as f:
    f.write(str(ds) + "\n\n")
    for v in ds.data_vars:
        f.write(f"{v}: dims={ds[v].dims} shape={ds[v].shape} attrs={dict(ds[v].attrs)}\n")
print(ds)

lat_name = [d for d in ds.dims if d.lower().startswith("lat")][0]
lon_name = [d for d in ds.dims if d.lower().startswith("lon")][0]
time_name = "time" if "time" in ds.dims else [d for d in ds.dims if "time" in d.lower()][0]
others = [d for d in ds["u10"].dims if d not in (lat_name, lon_name, time_name)]
mem_name = others[0] if others else None
print("dims ->", dict(lat=lat_name, lon=lon_name, time=time_name, member=mem_name))

order = [d for d in (mem_name, time_name, lat_name, lon_name) if d is not None]
u = ds["u10"].transpose(*order)
v = ds["v10"].transpose(*order)

lats = ds[lat_name].values
lons = ds[lon_name].values
i0 = int(np.argmin(np.abs(lats - LAT0)))
j0 = int(np.argmin(np.abs(lons - LON0)))
isl = slice(max(i0 - HALF, 0), min(i0 + HALF + 1, len(lats)))
jsl = slice(max(j0 - HALF, 0), min(j0 + HALF + 1, len(lons)))

dlat = float(np.abs(np.median(np.diff(lats))))
step = max(int(round(COARSE_DEG / dlat)), 1)

out = dict(
    times=ds[time_name].values.astype("datetime64[s]").astype(str),
    members=(ds[mem_name].values if mem_name else np.array([0])),
    site_lat=lats[isl], site_lon=lons[jsl],
    site_u10=u.isel({lat_name: isl, lon_name: jsl}).values.astype(np.float32),
    site_v10=v.isel({lat_name: isl, lon_name: jsl}).values.astype(np.float32),
    coarse_lat=lats[::step], coarse_lon=lons[::step],
    coarse_u10=u.isel({lat_name: slice(None, None, step), lon_name: slice(None, None, step)}).values.astype(np.float32),
    coarse_v10=v.isel({lat_name: slice(None, None, step), lon_name: slice(None, None, step)}).values.astype(np.float32),
    grid_spacing_deg=np.array(dlat),
    dim_order=np.array(order),
)
# keep any other site-only variables, if present
for name in ds.data_vars:
    if name in ("u10", "v10"):
        continue
    da = ds[name]
    if lat_name in da.dims and lon_name in da.dims:
        out["site_" + name] = da.isel({lat_name: isl, lon_name: jsl}).values.astype(np.float32)

np.savez_compressed(OUT / "cass_wind_subset.npz", **out)
print("Saved", OUT / "cass_wind_subset.npz", "with keys:", list(out))
print("site block shape:", out["site_u10"].shape, " coarse grid shape:", out["coarse_u10"].shape)
