"""How many AQ readings fall outside working hours? Clock-based and GPS-movement-based splits,
restricted to roster rider-node assignment periods."""
import _paths  # side effect: chdir into data/review_2026_09/
import sys, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import build_pvt_aq_dataset as B

m = pd.read_csv("aq_minute.csv", usecols=["node", "minute", "n", "disp_m", "pm25"])
t = pd.to_datetime(m.minute, unit="s", utc=True).dt.tz_convert("Asia/Bangkok")
m["date"] = t.dt.date; m["hr"] = t.dt.hour + t.dt.minute / 60
m["moving"] = m.disp_m.between(100, 2000).astype(int)

# keep roster assignments only
ro = B.load_roster()
keep = []
for r in ro.itertuples():
    g = m[(m.node == r.Node) & (m.date >= r._3.date()) & (m.date <= r._4.date())].copy()
    g["rider"] = r._1
    keep.append(g)
m = pd.concat(keep)
# drop the 1-reading export-boundary days
day_n = m.groupby(["rider", "date"]).n.transform("sum")
m = m[day_n >= 3600]
tot = m.n.sum()
print(f"Roster riders' AQ readings: {tot:,} (~1 Hz) over {m.groupby(['rider','date']).ngroups} rider-days, "
      f"{m.date.nunique()} distinct study days; weekday share: {100*np.mean(pd.to_datetime(m.date.astype(str)).dt.dayofweek < 5):.1f}%")

# (a) clock-based
for lo, hi in ((7, 19), (6, 20)):
    on = m.hr.between(lo, hi, inclusive="left")
    print(f"Clock {lo:02d}:00-{hi:02d}:00 as working: off-hours readings {m.n[~on].sum():,} ({100*m.n[~on].sum()/tot:.1f}%)")

# (b) GPS-movement-based shift span per rider-day
def span(g):
    g = g.sort_values("minute")
    mv = g.moving.to_numpy()
    fwd = pd.Series(mv[::-1]).rolling(10, min_periods=1).sum().to_numpy()[::-1]
    bwd = pd.Series(mv).rolling(10, min_periods=1).sum().to_numpy()
    s = np.where((mv == 1) & (fwd >= 3) & (g.hr.to_numpy() >= 4))[0]
    e = np.where((mv == 1) & (bwd >= 3))[0]
    if not len(s) or not len(e) or e[-1] <= s[0]:
        return pd.Series({"start": np.nan, "end": np.nan})
    return pd.Series({"start": g.hr.iloc[s[0]], "end": g.hr.iloc[e[-1]]})
sp = m.groupby(["rider", "date"]).apply(span).reset_index()
m = m.merge(sp, on=["rider", "date"])
has = m.start.notna()
on = has & (m.hr >= m.start) & (m.hr <= m.end)
print(f"\nGPS-movement shift detected on {sp.start.notna().sum()}/{len(sp)} rider-days; "
      f"median shift {sp.start.median():.1f}h -> {sp.end.median():.1f}h (IQR start {sp.start.quantile(.25):.1f}-{sp.start.quantile(.75):.1f}, "
      f"end {sp.end.quantile(.25):.1f}-{sp.end.quantile(.75):.1f})")
print(f"  on-shift readings   {m.n[on].sum():>11,} ({100*m.n[on].sum()/tot:.1f}%)")
print(f"  off-shift readings  {m.n[has & ~on].sum():>11,} ({100*m.n[has & ~on].sum()/tot:.1f}%)")
print(f"  no movement that day {m.n[~has].sum():>10,} ({100*m.n[~has].sum()/tot:.1f}%)  <- sensor likely not carried")
v = m.pm25.notna()
print(f"  PM2.5 mean on-shift {np.average(m.pm25[on & v], weights=m.n[on & v]):.1f}  off-shift {np.average(m.pm25[has & ~on & v], weights=m.n[has & ~on & v]):.1f} ug/m3; on-shift h/rider-day {m.n[on].sum()/3600/has.groupby([m.rider, m.date]).any().sum():.1f}")
per = m.assign(off=(has & ~on) * m.n, onn=on * m.n).groupby("rider")[["off", "onn"]].sum()
per["off_h_per_day"] = per.off / 3600 / m.groupby("rider").date.nunique()
per["pct_off"] = 100 * per.off / (per.off + per.onn)
print("\nPer rider (movement-based):")
print(per.assign(off=per.off.map("{:,}".format)).round(1).sort_values("pct_off").to_string())
