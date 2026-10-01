"""POST-HOC: pair each study day with its adjacent study day(s) for the same rider."""
import _paths  # side effect: chdir into data/review_2026_09/
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, statsmodels.formula.api as smf

c = pd.read_csv("linked_corrected.csv").dropna(subset=["response_speed_hz", "pm25_mean_sameday"])
met = {}
for reg, st in (("Bangkok", "VTBD"), ("Chiang mai", "VTCC")):
    d = pd.read_csv(f"metar_{st}.csv", na_values="M").dropna(subset=["tmpc"])
    d["t"] = pd.to_datetime(d.valid).dt.tz_localize("Asia/Bangkok").astype("int64") // 10**9
    met[reg] = d.sort_values("t")[["t", "tmpc"]].to_numpy()
c["t_amb"] = [np.interp(t, met[r][:, 0], met[r][:, 1]) for r, t in zip(c.region, c.ts_test)]
day = (c.groupby(["region", "username", "study_day"])
       .agg(speed=("response_speed_hz", "mean"), pm=("pm25_mean_sameday", "mean"), post=("pm25_mean_post4h", "mean"),
            hour=("hour_of_day", "mean"), tamb=("t_amb", "mean"), n=("response_speed_hz", "size")).reset_index())
day["d"] = pd.to_datetime(day.study_day)
S = c.response_speed_hz.mean()
ms = lambda b: 1000 / (S + b * 50) - 1000 / S

# A. first differences between consecutive study days
fd = []
for u, g in day.sort_values("d").groupby("username"):
    g = g.reset_index(drop=True)
    for i in range(1, len(g)):
        a, b = g.loc[i - 1], g.loc[i]
        gap = (b.d - a.d).days
        if gap > 7: continue
        fd.append({"username": u, "region": b.region, "study_day": b.study_day, "gap": gap,
                   "dy": b.speed - a.speed, "dpm": b.pm - a.pm, "dpost": b.post - a.post, "dhour": b.hour - a.hour, "dT": b.tamb - a.tamb})
fd = pd.DataFrame(fd)
# B. symmetric: day vs linear interpolation of the neighbours on both sides
sy = []
for u, g in day.sort_values("d").groupby("username"):
    g = g.reset_index(drop=True)
    for i in range(1, len(g) - 1):
        p, x, n = g.loc[i - 1], g.loc[i], g.loc[i + 1]
        if (x.d - p.d).days > 7 or (n.d - x.d).days > 7: continue
        w = (x.d - p.d).days / (n.d - p.d).days
        dev = lambda col: x[col] - (p[col] + w * (n[col] - p[col]))
        sy.append({"username": u, "region": x.region, "study_day": x.study_day,
                   "dy": dev("speed"), "dpm": dev("pm"), "dpost": dev("post"), "dhour": dev("hour"), "dT": dev("tamb")})
sy = pd.DataFrame(sy)

def fit(label, f, d, term="dpm"):
    d = d.dropna(subset=["dy", "dpm", "dhour", "dT"] + (["dpost"] if "dpost" in f else []))
    g = np.column_stack([pd.factorize(d.username)[0], pd.factorize(d.study_day)[0]])
    r = smf.ols(f, d).fit(cov_type="cluster", cov_kwds={"groups": g})
    b, (lo, hi) = r.params[term], r.conf_int().loc[term]
    print(f"  {label:58s} n={len(d):4d}  beta={b*1e4:+6.2f} [{lo*1e4:+6.2f}, {hi*1e4:+6.2f}]  p={r.pvalues[term]:.3f}  "
          f"worst case {ms(lo):+.1f} ms per +50")
    return r

print(f"A. Consecutive study days (<=7 d apart): {len(fd)} pairs, median gap {fd.gap.median():.0f} d, "
      f"|dPM2.5| median {fd.dpm.abs().median():.0f}, SD {fd.dpm.std():.0f} ug/m3")
ra = fit("change in pre-test PM2.5", "dy ~ dpm + dhour + dT + gap", fd)
print(f"     intercept (average drift per step) = {ra.params['Intercept']:+.4f} Hz, gap term {ra.params['gap']:+.4f} Hz/day")
fit("  joint: pre-test term", "dy ~ dpm + dpost + dhour + dT + gap", fd)
fit("  joint: PLACEBO post-test term", "dy ~ dpm + dpost + dhour + dT + gap", fd, "dpost")
fd["cd"] = fd.region + fd.study_day
fit("  + same-city same-day FE", "dy ~ dpm + dhour + dT + gap + C(cd)", fd)

print(f"\nB. Symmetric (day vs interpolated neighbours): {len(sy)} rider-days, dev PM2.5 SD {sy.dpm.std():.0f} ug/m3")
fit("deviation in pre-test PM2.5", "dy ~ dpm + dhour + dT", sy)
fit("  joint: pre-test term", "dy ~ dpm + dpost + dhour + dT", sy)
fit("  joint: PLACEBO post-test term", "dy ~ dpm + dpost + dhour + dT", sy, "dpost")
sy["cd"] = sy.region + sy.study_day
fit("  + same-city same-day FE", "dy ~ dpm + dhour + dT + C(cd)", sy)
fit("  + same-city same-day FE, joint: PLACEBO term", "dy ~ dpm + dpost + dhour + dT + C(cd)", sy, "dpost")
print(f"   share of dev-PM2.5 variance left after same-day FE: {smf.ols('dpm ~ C(cd)', sy).fit().resid.var()/sy.dpm.var():.2f}")
print(f"   ambient temperature term (symmetric, no FE): see below")
r = smf.ols("dy ~ dpm + dhour + dT", sy).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(sy.study_day)[0]})
print(f"   dT = {r.params['dT']:+.4f} Hz per C (p={r.pvalues['dT']:.3f})")

print("\nAmbient temperature term across designs (Hz per C; ms per C at mean speed):")
for lab, f, d in (("A consecutive days", "dy ~ dpm + dhour + dT + gap", fd), ("B symmetric", "dy ~ dpm + dhour + dT", sy),
                  ("B symmetric, no PM2.5 term", "dy ~ dhour + dT", sy)):
    dd = d.dropna(subset=["dy", "dpm", "dhour", "dT"])
    g = np.column_stack([pd.factorize(dd.username)[0], pd.factorize(dd.study_day)[0]])
    r = smf.ols(f, dd).fit(cov_type="cluster", cov_kwds={"groups": g})
    lo, hi = r.conf_int().loc["dT"]
    print(f"  {lab:28s} {r.params['dT']:+.4f} [{lo:+.4f}, {hi:+.4f}]  p={r.pvalues['dT']:.4f}  = {1000/(S + r.params['dT']) - 1000/S:+.1f} ms per C"
          f"  (dT SD {dd.dT.std():.1f} C)")
