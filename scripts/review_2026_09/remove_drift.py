"""POST-HOC: remove the effort/performance drift explicitly and see what PM2.5 signal is left."""
import _paths  # side effect: chdir into data/review_2026_09/
import ast, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, statsmodels.formula.api as smf

c = pd.read_csv("linked_corrected.csv")
o = pd.read_csv(f"{_paths.DATA}/pvt_aq_linked.csv", usecols=["timestamp", "username", "response_times"])
assert (c.timestamp.values == o.timestamp.values).all()
rt = o.response_times.apply(lambda s: np.array(ast.literal_eval(s), dtype=float))
c["fast10_speed"] = rt.apply(lambda r: 1000 / np.percentile(r[r >= 100], 10))    # optimal-response speed, less effort-sensitive
c["sdn"] = (pd.to_datetime(c.study_day) - pd.Timestamp("2025-12-12")).dt.days
c["week"] = pd.to_datetime(c.study_day).dt.isocalendar().week.astype(str)
c["rider_week"] = c.username + "_" + c.week
c["region_day"] = c.region + "_" + c.study_day
cov = "temp_mean_sameday + humidity_mean_sameday + session_number + hour_of_day"
d0 = c.dropna(subset=["response_speed_hz", "pm25_mean_sameday", "pm25_mean_post4h", "temp_mean_sameday", "humidity_mean_sameday"]).copy()
d0["day_code"] = pd.factorize(d0.study_day)[0]

designs = {
    "A  rider FE only (drift left in)":                  "C(username)",
    "B  + each rider's own smooth trend (spline, 4 df)": "C(username) + C(username):cr(sdn, df=4)",
    "C  + each rider's own week FE":                      "C(rider_week)",
    "D  same city, same day FE (shared drift removed)":   "C(username) + C(region_day)",
    "E  D + each rider's own week FE":                    "C(rider_week) + C(region_day)",
}
print("Share of within-rider PM2.5 variance left after removing drift, and pooled estimates (x1e-4, day-clustered):")
base_var = smf.ols("pm25_mean_sameday ~ C(username)", d0).fit().resid.var()
for lab, fe in designs.items():
    left = smf.ols(f"pm25_mean_sameday ~ {fe}", d0).fit().resid.var() / base_var
    out = [f"{lab:52s} PM var left {100*left:3.0f}% |"]
    for y, yl in (("response_speed_hz", "mean speed"), ("fast10_speed", "fastest-10%")):
        for x, xl in (("pm25_mean_sameday", "pre"), ("pm25_mean_post4h", "PLACEBO")):
            f = smf.ols(f"{y} ~ {x} + {cov} + {fe}", d0).fit(cov_type="cluster", cov_kwds={"groups": d0.day_code})
            lo, hi = f.conf_int().loc[x]
            star = "*" if f.pvalues[x] < .05 else " "
            out.append(f" {yl} {xl}: {f.params[x]*1e4:+5.1f} [{lo*1e4:+5.1f},{hi*1e4:+5.1f}]{star}")
    print(out[0]); print("     " + " ".join(out[1:3])); print("     " + " ".join(out[3:5]))
# how much of the drift does each design remove? (session_number coefficient in the mean-speed model)
print("\nDrift left in the outcome (rider-demeaned speed, sessions 1-10 minus sessions 80+), after each design's FE:")
for lab, fe in designs.items():
    r = smf.ols(f"response_speed_hz ~ {fe}", d0).fit().resid
    print(f"  {lab:52s} {r[d0.session_number <= 10].mean() - r[d0.session_number > 80].mean():+.3f} Hz")
