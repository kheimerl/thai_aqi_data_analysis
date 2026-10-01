"""POST-HOC: why do the regional PM2.5 slopes have opposite signs?"""
import _paths  # side effect: chdir into data/review_2026_09/
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, statsmodels.formula.api as smf
from run_reanalysis import per_rider, pool, fmt
df = pd.read_csv("linked_corrected.csv")
df["study_day_num"] = (pd.to_datetime(df.study_day) - pd.Timestamp("2025-12-12")).dt.days
df["log_session"] = np.log(df.session_number)
Y = "response_speed_hz"
start = df.groupby("username").study_day.min()
print("Rider start dates by region:", df.drop_duplicates("username").assign(s=lambda d: d.username.map(start)).groupby("region").s.value_counts().to_dict())

# 1. Which way does each rider's PM2.5 trend over their own testing history?
c = df.dropna(subset=["pm25_mean_sameday"]).groupby(["region", "username"]).apply(
    lambda g: pd.Series({"r_pm_session": g.pm25_mean_sameday.corr(g.session_number)})).reset_index()
print("\nWithin-rider corr(PM2.5, session_number):")
print(c.groupby("region").r_pm_session.agg(["mean", "min", "max", lambda s: int((s > 0).sum()), "size"]).round(2).to_string())

# 2. Shape of the performance decline: residual speed by session bin (rider-demeaned, pooled)
d = df.dropna(subset=[Y]).copy()
d["dm"] = d[Y] - d.groupby("username")[Y].transform("mean")
bins = pd.cut(d.session_number, [0, 10, 20, 40, 60, 80, 100, 250])
print("\nRider-demeaned speed (Hz) by session bin:", d.groupby(bins).dm.mean().round(3).to_dict())

# 3. Per-region pooled estimate under progressively stronger time controls
base = ["temp_mean_sameday", "humidity_mean_sameday", "shift_duration_min_sameday", "hour_of_day"]
specs = {
    "linear session (plan of record)": ["session_number"],
    "+ log(session) (nonlinear decline)": ["session_number", "log_session"],
    "+ linear study_day": ["session_number", "log_session", "study_day_num"],
}
print("\nPer-rider R1 pooling by region, x1e-4:")
for lab, tc in specs.items():
    t = per_rider(df, Y, ["pm25_mean_sameday"], base + tc).assign(region=lambda x: x.username.map(df.drop_duplicates("username").set_index("username").region))
    for reg in ("Bangkok", "Chiang mai"):
        s = t[t.region == reg]
        print(fmt(f"  {lab:36s} {reg}", pool(s.b_pm25_mean_sameday, s.se_pm25_mean_sameday)))
# 4. Within-region, rider FE + day FE (only same-city same-day comparisons)
print("\nWithin region, pooled rider FE + day FE (day-clustered):")
for reg, g in df.dropna(subset=[Y, "pm25_mean_sameday", "temp_mean_sameday", "humidity_mean_sameday"]).groupby("region"):
    for lab, f in (("rider FE only", "C(username)"), ("rider FE + day FE", "C(username) + C(study_day)")):
        fit = smf.ols(f"{Y} ~ pm25_mean_sameday + temp_mean_sameday + humidity_mean_sameday + session_number + hour_of_day + {f}", g).fit(
            cov_type="cluster", cov_kwds={"groups": pd.factorize(g.study_day)[0]})
        lo, hi = fit.conf_int().loc["pm25_mean_sameday"]
        print(f"  {reg:10s} {lab:18s} beta={fit.params['pm25_mean_sameday']*1e4:+6.2f}  95%CI=[{lo*1e4:+6.2f}, {hi*1e4:+6.2f}]  p={fit.pvalues['pm25_mean_sameday']:.3f}  n={len(g)}")
