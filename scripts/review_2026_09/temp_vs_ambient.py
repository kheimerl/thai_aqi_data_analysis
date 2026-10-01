"""POST-HOC: does the sensor's temperature track ambient air temperature?"""
import _paths  # side effect: chdir into data/review_2026_09/
import json, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd

amb = []
for reg, f in (("Bangkok", "era5_bangkok.json"), ("Chiang mai", "era5_chiangmai.json")):
    h = json.load(open(f))["hourly"]
    amb.append(pd.DataFrame({"region": reg, "hour": pd.to_datetime(h["time"]), "era5": h["temperature_2m"], "era5_rh": h["relative_humidity_2m"]}))
amb = pd.concat(amb)
met = []
for reg, st in (("Bangkok", "VTBD"), ("Chiang mai", "VTCC")):
    d = pd.read_csv(f"metar_{st}.csv", na_values="M")
    d["hour"] = pd.to_datetime(d.valid).dt.floor("h")
    met.append(d.groupby("hour").agg(metar=("tmpc", "mean"), metar_rh=("relh", "mean")).reset_index().assign(region=reg))
amb = amb.merge(pd.concat(met), on=["region", "hour"], how="left")
print("ERA5 vs METAR hourly agreement:", {r: round(g[["era5", "metar"]].corr().iloc[0, 1], 3) for r, g in amb.groupby("region")},
      " mean ERA5-METAR (C):", {r: round((g.era5 - g.metar).mean(), 2) for r, g in amb.groupby("region")})

m = pd.read_csv("minute_classes.csv")
m["hour"] = pd.to_datetime(m.minute, unit="s", utc=True).dt.tz_convert("Asia/Bangkok").dt.tz_localize(None).dt.floor("h")
rh_ = (m.dropna(subset=["temp"]).groupby(["rider", "region", "cls", "hour"])
       .apply(lambda g: pd.Series({"temp": np.average(g.temp, weights=g.n), "rh": np.average(g.rh.fillna(g.rh.mean()), weights=g.n), "mins": len(g)}))
       .reset_index())
rh_ = rh_[rh_.mins >= 15].merge(amb, on=["region", "hour"], how="left")
rh_["date"] = rh_.hour.dt.date; rh_["h"] = rh_.hour.dt.hour
rh_.to_csv("temp_rider_hours.csv", index=False)

def summ(g, ref):
    g = g.dropna(subset=[ref])
    r_all = g[["temp", ref]].corr().iloc[0, 1]
    dm = g.groupby(["rider", "date"])[["temp", ref]].mean()           # day-to-day (daily means)
    dmr = dm - dm.groupby(level="rider").transform("mean")
    r_day = dmr.corr().iloc[0, 1]
    wd = g[["temp", ref]] - g.groupby(["rider", "date"])[["temp", ref]].transform("mean")   # within-day (diurnal)
    r_diur = wd.corr().iloc[0, 1]
    slope = np.polyfit(g[ref], g.temp, 1)[0]
    return dict(n_rider_hours=len(g), r_hourly=round(r_all, 2), r_daily_means=round(r_day, 2), r_within_day=round(r_diur, 2),
                bias_C=round((g.temp - g[ref]).mean(), 1), slope=round(slope, 2))

for ref in ("metar", "era5"):
    print(f"\n=== sensor temp vs {ref.upper()} ===")
    out = {(c, r): summ(g, ref) for (c, r), g in rh_[rh_.cls.isin(["road", "home", "other stationary"])].groupby(["cls", "region"])}
    print(pd.DataFrame(out).T.to_string())

print("\nMean sensor-minus-METAR (C) by clock hour:")
p = rh_[rh_.cls.isin(["road", "home"])].assign(bias=lambda d: d.temp - d.metar).pivot_table(index="cls", columns="h", values="bias", aggfunc="mean").round(1)
print(p.to_string())
print("\nMean temps by hour, road minutes (sensor vs METAR), both regions pooled:")
q = rh_[rh_.cls == "road"].groupby("h")[["temp", "metar"]].mean().round(1).T
print(q.to_string())
