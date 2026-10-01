"""POST-HOC descriptive: is PM2.5 measured at home distinguishable from PM2.5 on the road?
Road = minutes with 100-2000 m GPS displacement. Home = stationary within 150 m of the rider's
nightly (01:00-04:00) modal location. Everything else with GPS = other stationary (stands, stops)."""
import _paths  # side effect: chdir into data/review_2026_09/
import sys, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import build_pvt_aq_dataset as B

m = pd.read_csv("aq_minute.csv")
t = pd.to_datetime(m.minute, unit="s", utc=True).dt.tz_convert("Asia/Bangkok")
m["date"] = t.dt.date; m["hr"] = t.dt.hour + t.dt.minute / 60; m["h"] = t.dt.hour
ro = B.load_roster()
reg = pd.read_excel(f"{_paths.ROOT}/2025-2026 Rider-Sensor-Summary.xlsx", sheet_name="summary")
reg = dict(zip(reg["PVT user"].astype(str).str.strip(), reg.Region))
parts = []
for r in ro.itertuples():
    g = m[(m.node == r.Node) & (m.date >= r._3.date()) & (m.date <= r._4.date())].copy()
    g["rider"] = r._1; g["region"] = reg[r._1]
    parts.append(g)
m = pd.concat(parts)
m = m[m.groupby(["rider", "date"]).n.transform("sum") >= 3600]

# home location: rounded-grid mode of 01:00-04:00 fixes
night = m[(m.h >= 1) & (m.h < 4) & m.lat.notna()]
home = {}
for rid, g in night.groupby("rider"):
    key = (g.lat.round(3).astype(str) + "," + g.lon.round(3).astype(str)).mode().iloc[0]
    la, lo = map(float, key.split(","))
    near = g[(np.hypot((g.lat - la) * 111e3, (g.lon - lo) * 111e3 * np.cos(np.radians(la))) < 300)]
    home[rid] = (near.lat.median(), near.lon.median(), len(near) / len(g))
hl = m.rider.map(lambda r: home.get(r, (np.nan,) * 3))
m["home_lat"], m["home_lon"] = [x[0] for x in hl], [x[1] for x in hl]
m["d_home"] = np.hypot((m.lat - m.home_lat) * 111e3, (m.lon - m.home_lon) * 111e3 * np.cos(np.radians(m.home_lat)))
moving = m.disp_m.between(100, 2000)
m["cls"] = np.select([moving, (m.d_home < 150) & ~moving, m.lat.notna()], ["road", "home", "other stationary"], "no GPS")
print("Share of 01-04h fixes within 300 m of the nightly mode, per rider (home-location stability):",
      f"median {np.median([v[2] for v in home.values()]):.2f}, min {min(v[2] for v in home.values()):.2f}")

def wavg(x, w):
    k = x.notna(); return np.average(x[k], weights=w[k])
rows = []
for c, g in m.groupby("cls"):
    rows.append({"class": c, "hours": g.n.sum() / 3600, "PM2.5 mean": wavg(g.pm25, g.n), "median": g.pm25.median(),
                 "p90": g.pm25.quantile(.9), "RH mean": wavg(g.rh, g.n), "temp mean": wavg(g.temp, g.n),
                 "min-to-min |dPM|": g.sort_values(["rider", "minute"]).groupby("rider").pm25.diff().abs().median()})
print("\nPooled, all riders/days:")
print(pd.DataFrame(rows).set_index("class").round(1).to_string())

# paired rider-day comparison
rd = (m[m.cls.isin(["road", "home"])].groupby(["rider", "region", "date", "cls"])
      .apply(lambda g: pd.Series({"pm": wavg(g.pm25, g.n), "min": len(g)})).unstack("cls"))
rd = rd[(rd[("min", "road")] >= 60) & (rd[("min", "home")] >= 60)]
diff = rd[("pm", "road")] - rd[("pm", "home")]
print(f"\nPaired rider-days with >=60 min each of road and home: {len(rd)}")
print(f"  road - home: median {diff.median():+.1f} ug/m3, IQR [{diff.quantile(.25):+.1f}, {diff.quantile(.75):+.1f}], road higher on {100*(diff>0).mean():.0f}% of rider-days")
for rg, d in diff.groupby(level="region"):
    print(f"  {rg}: median {d.median():+.1f}, road higher on {100*(d>0).mean():.0f}% (n={len(d)})")
# same-hours comparison (removes diurnal / boundary-layer difference): hours where both classes occur on that rider-day
hh = m[m.cls.isin(["road", "home"])].groupby(["rider", "date", "h", "cls"]).pm25.mean().unstack("cls").dropna()
dh = hh.road - hh.home
print(f"  same rider-day AND same clock hour ({len(hh)} rider-day-hours): median road-home {dh.median():+.1f}, road higher {100*(dh>0).mean():.0f}%")
lowrh = m[m.rh < 70]
hh2 = lowrh[lowrh.cls.isin(["road", "home"])].groupby(["rider", "date", "h", "cls"]).pm25.mean().unstack("cls").dropna()
print(f"  same as above, RH<70% only ({len(hh2)}): median road-home {(hh2.road-hh2.home).median():+.1f}")
# do home and road track each other day to day?
pm = rd["pm"]
dm = pm.groupby(level="rider").transform(lambda s: s - s.mean())
print(f"  within-rider day-to-day corr(home daily mean, road daily mean) = {dm.road.corr(dm.home):.2f}")
print(f"  between-rider corr of rider means: {pm.groupby(level='rider').mean().corr().iloc[0,1]:.2f}")

print("\nHourly PM2.5 by class (all riders), ug/m3:")
prof = m[m.cls.isin(["road", "home", "other stationary"])].groupby(["h", "cls"]).pm25.mean().unstack("cls").round(0)
print(prof.T.to_string())
print("\nHourly mean RH (%) at home vs road:")
print(m[m.cls.isin(["road", "home"])].groupby(["h", "cls"]).rh.mean().unstack("cls").round(0).T.to_string())
m[["rider", "region", "date", "minute", "cls", "pm25", "rh", "temp", "n"]].to_csv("minute_classes.csv", index=False)
