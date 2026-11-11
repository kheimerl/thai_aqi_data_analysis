"""One chunked pass over data/aq_sensor.csv -> per-node per-minute aggregates.

Output: foundation/_work/aq_min/<node>.csv  (derived, large-ish; not for sharing)
Columns: minute_unix (Unix s, floor to 60), n_rows, n_pm (PM in [0,500]), pm_sum,
 rh_sum, n_rh, t_sum, n_t, lat_sum, lon_sum, n_gps, ts_mismatch (rows where
 Datetime(UTC+7) != Timestamp + 7 h).
"""
import os, sys, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from foundation_common import DATA, FOUND

OUT = os.path.join(FOUND, "_work", "aq_min"); os.makedirs(OUT, exist_ok=True)
cols = ["Timestamp", "Node", "Datetime(UTC+7)", "GPS_Lat", "GPS_Lon", "Relative humidity", "PM2.5", "Temperature"]
parts = {}
t0 = time.time(); nrows = 0; bad_ts = 0; check_rows = 0
for i, ch in enumerate(pd.read_csv(os.path.join(DATA, "aq_sensor.csv"), usecols=cols, chunksize=4_000_000,
                                   dtype={"Node": str, "Datetime(UTC+7)": str})):
    nrows += len(ch)
    ts = pd.to_numeric(ch["Timestamp"], errors="coerce")
    ch = ch[ts.notna()]; ts = ts[ts.notna()].astype(np.int64)
    # clock check on every 50th row: Datetime(UTC+7) must equal Timestamp + 7h
    samp = ch.iloc[::50]
    dt = pd.to_datetime(samp["Datetime(UTC+7)"], errors="coerce")
    exp = pd.Timestamp("1970-01-01") + pd.to_timedelta(ts.iloc[::50].values + 7 * 3600, unit="s")
    bad_ts += int((dt.values != exp.values).sum()); check_rows += len(samp)
    pm = pd.to_numeric(ch["PM2.5"], errors="coerce"); pm = pm.where((pm >= 0) & (pm <= 500))
    rh = pd.to_numeric(ch["Relative humidity"], errors="coerce")
    tt = pd.to_numeric(ch["Temperature"], errors="coerce")
    lat = pd.to_numeric(ch["GPS_Lat"], errors="coerce"); lon = pd.to_numeric(ch["GPS_Lon"], errors="coerce")
    g = (lat.abs() > 1) & (lon.abs() > 1) & lat.between(5, 21) & lon.between(97, 106)  # Thailand box, drop 0/0
    d = pd.DataFrame({"node": ch["Node"].str.replace("Grab-", "", regex=False).values,
                      "minute_unix": (ts.values // 60) * 60, "n_rows": 1,
                      "n_pm": pm.notna().astype(int).values, "pm_sum": pm.fillna(0).values,
                      "n_rh": rh.notna().astype(int).values, "rh_sum": rh.fillna(0).values,
                      "n_t": tt.notna().astype(int).values, "t_sum": tt.fillna(0).values,
                      "n_gps": g.astype(int).values, "lat_sum": lat.where(g, 0).values, "lon_sum": lon.where(g, 0).values})
    agg = d.groupby(["node", "minute_unix"], sort=False).sum()
    for node, sub in agg.groupby(level=0):
        parts.setdefault(node, []).append(sub.droplevel(0))
    print(f"chunk {i} rows {nrows:,} elapsed {time.time()-t0:.0f}s", flush=True)

for node, lst in parts.items():
    a = pd.concat(lst).groupby(level=0).sum().sort_index()
    a.to_csv(os.path.join(OUT, f"{node}.csv"))
print("TOTAL rows", nrows, "nodes", sorted(parts), "clock-check rows", check_rows, "mismatches", bad_ts)
