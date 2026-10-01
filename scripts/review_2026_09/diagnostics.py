"""POST-HOC: effect-size scale, per-rider original vs corrected slopes, and the session_number decline
(calendar vs testing history, by region)."""
import _paths  # side effect: chdir into data/review_2026_09/
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, statsmodels.formula.api as smf
from scipy import stats
from run_reanalysis import per_rider, pool
df = pd.read_csv("linked_corrected.csv")
df["study_day_num"] = (pd.to_datetime(df.study_day) - pd.Timestamp("2025-12-12")).dt.days
df["month"] = pd.to_datetime(df.study_day).dt.to_period("M").astype(str)
Y = "response_speed_hz"

# 1. Effect-size scale
w_sd = df.groupby("username")[Y].transform(lambda s: s - s.mean()).std()
pm_sd = df.groupby("username").pm25_mean_sameday.transform(lambda s: s - s.mean()).std()
mean_speed = df[Y].mean()
for lab, b in (("original claim", -6.41e-4), ("corrected R1", -2.05e-4)):
    d_hz = b * pm_sd
    d_ms = 1000 / (mean_speed + d_hz) - 1000 / mean_speed
    print(f"{lab}: per 1 within-rider SD of PM2.5 ({pm_sd:.0f} ug/m3): {d_hz:+.4f} Hz = {d_hz / w_sd:+.3f} within-rider SD of speed ~ {d_ms:+.1f} ms mean-RT-equivalent")

# 2. Per-rider corrected slopes vs original
orig = pd.read_csv(f"{_paths.SUPERSEDED}/primary_analysis_per_rider.csv")
t = per_rider(df, Y, ["pm25_mean_sameday"], ["temp_mean_sameday", "humidity_mean_sameday", "shift_duration_min_sameday", "session_number", "hour_of_day"])
m = orig[["username", "beta_pm25", "se_pm25"]].merge(t, on="username")
m["z_orig"] = m.beta_pm25 / m.se_pm25; m["z_corr"] = m.b_pm25_mean_sameday / m.se_pm25_mean_sameday
print("\nPer-rider z, original vs corrected (flagged riders from the precision check):")
print(m[["username", "z_orig", "z_corr"]].round(2).sort_values("z_orig").to_string(index=False))
print(f"corr(z_orig, z_corr) = {np.corrcoef(m.z_orig, m.z_corr)[0,1]:.2f}; corr(beta, se) corrected = {np.corrcoef(t.b_pm25_mean_sameday, t.se_pm25_mean_sameday)[0,1]:.2f}")

# 3. session_number decline: robust? calendar vs count
d = df.dropna(subset=[Y, "pm25_mean_sameday", "temp_mean_sameday", "humidity_mean_sameday"]).copy()
d["region_day"] = d.region + "_" + d.study_day.astype(str)
d["day_code"] = pd.factorize(d.study_day)[0]
for lab, f in (("rider FE", "C(username)"), ("rider FE + region x day FE", "C(username) + C(region_day)")):
    fit = smf.ols(f"{Y} ~ session_number + pm25_mean_sameday + temp_mean_sameday + humidity_mean_sameday + hour_of_day + {f}", d).fit(
        cov_type="cluster", cov_kwds={"groups": d.day_code})
    print(f"\nsession_number [{lab}]: {fit.params['session_number']:+.5f} Hz/session  p={fit.pvalues['session_number']:.4f}")
print("\nMonthly mean response speed by region (raw):")
print(df.pivot_table(index="month", columns="region", values=Y, aggfunc="mean").round(3).to_string())
print("\nMonthly mean corrected same-day PM2.5 and temp by region:")
print(df.pivot_table(index="month", columns="region", values=["pm25_mean_sameday", "temp_mean_sameday"], aggfunc="mean").round(1).to_string())
# per-region per-rider slope of speed on session_number (simple, within rider)
for reg, g in df.groupby("region"):
    sl = g.groupby("username").apply(lambda h: np.polyfit(h.session_number, h[Y], 1)[0] if len(h) > 15 else np.nan).dropna()
    tt = stats.ttest_1samp(sl, 0)
    print(f"{reg}: per-rider raw session slope mean {sl.mean():+.5f} Hz/session, {int((sl<0).sum())}/{len(sl)} negative, p={tt.pvalue:.4f}")
