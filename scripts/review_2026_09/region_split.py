"""POST-HOC (not in the spec): pool per-rider PM2.5 slopes separately by region."""
import _paths  # side effect: chdir into data/review_2026_09/
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from run_reanalysis import per_rider, pool, fmt
df = pd.read_csv("linked_corrected.csv")
reg = df.drop_duplicates("username").set_index("username").region
orig = pd.read_csv(f"{_paths.SUPERSEDED}/primary_analysis_per_rider.csv")
orig = orig[orig.included].assign(region=lambda d: d.username.map(reg))
cov = ["temp_mean_sameday", "humidity_mean_sameday", "shift_duration_min_sameday", "session_number", "hour_of_day"]
t = per_rider(df, "response_speed_hz", ["pm25_mean_sameday"], cov).assign(region=lambda d: d.username.map(reg))
tn = per_rider(df, "response_speed_hz", ["pm25_mean_post4h"],
               ["temp_mean_post4h", "humidity_mean_post4h", "shift_duration_min_sameday", "session_number", "hour_of_day"]).assign(region=lambda d: d.username.map(reg))
for r in ("Bangkok", "Chiang mai"):
    o = orig[orig.region == r]; c = t[t.region == r]; n = tn[tn.region == r]
    print(fmt(f"{r:10s} ORIGINAL buggy (DL+Wald)", pool(o.beta_pm25, o.se_pm25, "DL+Wald")),
          f"  riders neg {int((o.beta_pm25<0).sum())}/{len(o)}")
    print(fmt(f"{r:10s} corrected R1 (REML+HKSJ)", pool(c.b_pm25_mean_sameday, c.se_pm25_mean_sameday)),
          f"  riders neg {int((c.b_pm25_mean_sameday<0).sum())}/{len(c)}")
    print(fmt(f"{r:10s} corrected N1 placebo post-4h", pool(n.b_pm25_mean_post4h, n.se_pm25_mean_post4h)))
# Is the original heterogeneity a region split? Q_between on original estimates (fixed-effect subgroup test)
def fe(b, s):
    w = 1 / s**2; mu = np.sum(w*b)/np.sum(w); return mu, np.sum(w*(b-mu)**2), np.sum(w)
b, s = orig.beta_pm25.to_numpy(), orig.se_pm25.to_numpy()
_, q_tot, _ = fe(b, s)
q_within = sum(fe(orig[orig.region==r].beta_pm25.to_numpy(), orig[orig.region==r].se_pm25.to_numpy())[1] for r in ("Bangkok", "Chiang mai"))
from scipy import stats
qb = q_tot - q_within
print(f"\nORIGINAL: Q_total={q_tot:.1f} (df {len(b)-1}), Q_between regions={qb:.1f} (df 1, p={stats.chi2.sf(qb,1):.4f}), "
      f"Q_within={q_within:.1f} (df {len(b)-2}); I2 within regions = {max(0,(q_within-(len(b)-2))/q_within)*100:.1f}%")
