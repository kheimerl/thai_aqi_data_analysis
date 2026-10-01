"""Per-node, per-minute AQ summary (Bangkok time) incl. GPS displacement -> aq_minute.csv."""
import _paths  # side effect: chdir into data/review_2026_09/
import glob, os
import numpy as np, pandas as pd
from _paths import ROOT as REPO
parts = []
for f in sorted(glob.glob(f"{REPO}/data/aq_by_node/*.csv")):
    node = os.path.basename(f)[:-4]
    df = pd.read_csv(f, usecols=["Timestamp", "GPS_Lat", "GPS_Lon", "PM2.5", "Temperature", "Relative humidity"])
    df["minute"] = (df.Timestamp // 60) * 60
    g = df.groupby("minute").agg(n=("Timestamp", "size"), pm25=("PM2.5", "mean"),
                                 temp=("Temperature", "mean"), rh=("Relative humidity", "mean"),
                                 lat=("GPS_Lat", "median"), lon=("GPS_Lon", "median")).reset_index()
    g["node"] = node
    parts.append(g)
m = pd.concat(parts, ignore_index=True).sort_values(["node", "minute"])
# displacement to the next minute (m), only for consecutive minutes with fixes
nxt = m.groupby("node")[["minute", "lat", "lon"]].shift(-1)
ok = (nxt.minute - m.minute == 60)
dy = (nxt.lat - m.lat) * 111_000
dx = (nxt.lon - m.lon) * 111_000 * np.cos(np.radians(m.lat.fillna(15)))
m["disp_m"] = np.where(ok, np.hypot(dx, dy), np.nan)
m.to_csv("aq_minute.csv", index=False)
print(m.shape, m.disp_m.describe().round(1).to_dict())
