"""Per (node, Bangkok date): first/last GPS fix, AQ coverage. Cached to gps_days.csv."""
import _paths  # side effect: chdir into data/review_2026_09/
import glob, os
import numpy as np, pandas as pd
from _paths import ROOT as REPO
rows = []
for f in sorted(glob.glob(f"{REPO}/data/aq_by_node/*.csv")):
    node = os.path.basename(f)[:-4]
    df = pd.read_csv(f, usecols=["Timestamp", "GPS_Lat"])
    t = pd.to_datetime(df.Timestamp, unit="s", utc=True).dt.tz_convert("Asia/Bangkok")
    df["date"] = t.dt.date
    df["hr"] = t.dt.hour + t.dt.minute / 60 + t.dt.second / 3600
    for d, g in df.groupby("date"):
        if len(g) < 3600:
            continue
        gps = g[g.GPS_Lat.notna()]
        rows.append({"node": node, "date": d, "n": len(g),
                     "aq_first_hr": g.hr.min(), "aq_last_hr": g.hr.max(),
                     "gps_frac": len(gps) / len(g),
                     "gps_first_hr": gps.hr.min() if len(gps) else np.nan,
                     "gps_last_hr": gps.hr.max() if len(gps) else np.nan})
out = pd.DataFrame(rows)
out.to_csv("gps_days.csv", index=False)
print(out.groupby("node")[["gps_frac", "gps_first_hr", "gps_last_hr"]].median().round(2).to_string())
