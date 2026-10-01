"""POST-HOC within-day design: does the change in PVT between two tests on the same rider-day
track the air breathed between them?"""
import _paths  # side effect: chdir into data/review_2026_09/
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, statsmodels.formula.api as smf

s = pd.read_csv("linked_corrected.csv")
s = s[s.node.notna() & s.response_speed_hz.notna()].copy()
s["rd"] = s.username + "_" + s.study_day
Y = "response_speed_hz"

# 1. Where does the exposure variation live?
x = s.dropna(subset=["pm25_mean_sameday"])
tot = x.pm25_mean_sameday.var()
rider = x.groupby("username").pm25_mean_sameday.transform("mean")
rday = x.groupby("rd").pm25_mean_sameday.transform("mean")
print("Variance of the plan's exposure (midnight->test mean PM2.5):")
print(f"  between riders {100*(rider.var())/tot:.0f}%  |  between days within rider {100*((rday-rider).var())/tot:.0f}%  |  within rider-day {100*((x.pm25_mean_sameday-rday).var())/tot:.0f}%")
y = s[Y]; yr = s.groupby("username")[Y].transform("mean"); yd = s.groupby("rd")[Y].transform("mean")
print(f"Outcome (speed): between riders {100*yr.var()/y.var():.0f}%  |  between days {100*(yd-yr).var()/y.var():.0f}%  |  within rider-day {100*(y-yd).var()/y.var():.0f}%")

# 2. Interval exposure between consecutive tests (minute table, reading-weighted)
m = pd.read_csv("aq_minute.csv", usecols=["node", "minute", "n", "pm25"])
m["w"] = m.pm25.fillna(0) * m.n; m["nn"] = np.where(m.pm25.notna(), m.n, 0)
idx = {}
for node, g in m.sort_values("minute").groupby("node"):
    idx[node] = (g.minute.to_numpy(), np.r_[0, np.cumsum(g.w.to_numpy())], np.r_[0, np.cumsum(g.nn.to_numpy())])
def interval_mean(node, a, b):
    t, W, N = idx[node]; lo, hi = np.searchsorted(t, a), np.searchsorted(t, b)
    n = N[hi] - N[lo]; return (W[hi] - W[lo]) / n if n > 600 else np.nan   # need >=10 min of data
met = {}
for reg, st in (("Bangkok", "VTBD"), ("Chiang mai", "VTCC")):
    d = pd.read_csv(f"metar_{st}.csv", na_values="M"); d["t"] = pd.to_datetime(d.valid).dt.tz_localize("Asia/Bangkok").astype("int64") // 10**9
    met[reg] = d.dropna(subset=["tmpc"])[["t", "tmpc"]].sort_values("t").to_numpy()
def amb(reg, ts):
    a = met[reg]; return np.interp(ts, a[:, 0], a[:, 1])

s = s.sort_values(["rd", "ts_test"])
pairs = []
for rd, g in s.groupby("rd"):
    g = g.reset_index(drop=True)
    for i in range(1, len(g)):
        a, b = g.loc[i - 1], g.loc[i]
        gap_h = (b.ts_test - a.ts_test) / 3600
        if gap_h < 0.5:                      # back-to-back retests: no exposure contrast
            continue
        pairs.append({"rd": rd, "username": b.username, "region": b.region, "study_day": b.study_day,
                      "d_speed": b[Y] - a[Y], "d_speed_clean": b.response_speed_clean_hz - a.response_speed_clean_hz,
                      "gap_h": gap_h, "h0": a.hour_of_day, "h1": b.hour_of_day,
                      "pm_between": interval_mean(b.node, a.ts_test, b.ts_test),
                      "pm_before_first": interval_mean(b.node, a.ts_test - 4 * 3600, a.ts_test),   # air before the 1st test
                      "pm_after_second": interval_mean(b.node, b.ts_test, b.ts_test + 4 * 3600),    # placebo: air after the 2nd test
                      "d_temp_amb": amb(b.region, b.ts_test) - amb(b.region, a.ts_test)})
p = pd.DataFrame(pairs).dropna(subset=["pm_between"])
p["day_code"] = pd.factorize(p.study_day)[0]; p["region_day"] = p.region + p.study_day
print(f"\nConsecutive same-day test pairs >=30 min apart: {len(p)} pairs, {p.rd.nunique()} rider-days, {p.username.nunique()} riders; "
      f"median gap {p.gap_h.median():.1f} h")
print(f"  PM2.5 between tests: mean {p.pm_between.mean():.0f}, SD {p.pm_between.std():.0f}; "
      f"SD after region x day FE: {smf.ols('pm_between ~ C(region_day)', p).fit().resid.std():.0f} ug/m3")

base = "d_speed ~ {x} + gap_h + h0 + h1 + d_temp_amb + C(username)"
def show(label, f, data=p, term="pm_between"):
    fit = smf.ols(f, data).fit(cov_type="cluster", cov_kwds={"groups": data.day_code})
    lo, hi = fit.conf_int().loc[term]
    print(f"  {label:62s} beta={fit.params[term]*1e4:+6.2f}  95%CI=[{lo*1e4:+6.2f}, {hi*1e4:+6.2f}]  p={fit.pvalues[term]:.3f}")
print("\nChange in speed between consecutive same-day tests (x1e-4 Hz per ug/m3; negative = dirtier interval -> slower):")
show("air between tests", base.format(x="pm_between"))
show("air between tests, + region x day FE", base.format(x="pm_between") + " + C(region_day)")
show("air between tests, false starts removed", base.format(x="pm_between").replace("d_speed ~", "d_speed_clean ~"))
q = p.dropna(subset=["pm_after_second", "pm_before_first"])
show("PLACEBO: air in the 4h after the second test", base.format(x="pm_after_second"), q, "pm_after_second")
show("  joint: between-tests term (placebo + before-first in model)", base.format(x="pm_between + pm_after_second + pm_before_first"), q)
show("  joint: placebo term", base.format(x="pm_between + pm_after_second + pm_before_first"), q, "pm_after_second")

# morning vs evening only: first test before 12:00, a later test at/after 17:00
me = []
for rd, g in s.groupby("rd"):
    am, pm_ = g[g.hour_of_day < 12], g[g.hour_of_day >= 17]
    if len(am) and len(pm_):
        a, b = am.iloc[0], pm_.iloc[-1]
        me.append({"rd": rd, "username": a.username, "region": a.region, "study_day": a.study_day, "d_speed": b[Y] - a[Y],
                   "gap_h": (b.ts_test - a.ts_test) / 3600, "h0": a.hour_of_day, "h1": b.hour_of_day,
                   "pm_between": interval_mean(a.node, a.ts_test, b.ts_test), "d_temp_amb": amb(a.region, b.ts_test) - amb(a.region, a.ts_test)})
me = pd.DataFrame(me).dropna(subset=["pm_between"]); me["day_code"] = pd.factorize(me.study_day)[0]
print(f"\nMorning(<12h) -> evening(>=17h) pairs: {len(me)} rider-days, {me.username.nunique()} riders, mean change {me.d_speed.mean():+.3f} Hz")
show("morning->evening change vs shift-day air", base.format(x="pm_between"), me)

print("\nWith region x day FE (only same-city, same-day contrasts between riders' within-day changes):")
qq = q.assign(region_day=q.region + q.study_day)
show("PLACEBO + region x day FE", base.format(x="pm_after_second") + " + C(region_day)", qq, "pm_after_second")
show("joint + region x day FE: between-tests term", base.format(x="pm_between + pm_after_second + pm_before_first") + " + C(region_day)", qq)
show("joint + region x day FE: placebo term", base.format(x="pm_between + pm_after_second + pm_before_first") + " + C(region_day)", qq, "pm_after_second")
print(f"  corr(pm_between, pm_after_second) = {q.pm_between.corr(q.pm_after_second):.2f}; within region-day: "
      f"{smf.ols('pm_between ~ C(region_day)', qq).fit().resid.corr(smf.ols('pm_after_second ~ C(region_day)', qq).fit().resid):.2f}")

# Activity check: minutes spent moving (riding) in each window
mm = pd.read_csv("aq_minute.csv", usecols=["node", "minute", "disp_m"])
mm["mv"] = mm.disp_m.between(100, 2000).astype(int)
mv_idx = {node: (g.minute.to_numpy(), np.r_[0, np.cumsum(g.mv.to_numpy())]) for node, g in mm.sort_values("minute").groupby("node")}
def moving_h(node, a, b):
    t, C = mv_idx[node]; return (C[np.searchsorted(t, b)] - C[np.searchsorted(t, a)]) / 60
node_of = s.drop_duplicates("rd").set_index("rd").node
ts2 = s.sort_values(["rd", "ts_test"]).groupby("rd").ts_test.apply(list)
rows = []
for r in qq.itertuples():
    ts = ts2[r.rd]; node = node_of[r.rd]
    # recover this pair's test times from gap and the ordered list
    for i in range(1, len(ts)):
        if abs((ts[i] - ts[i - 1]) / 3600 - r.gap_h) < 1e-6:
            a, b = ts[i - 1], ts[i]; break
    rows.append({"mv_between_h": moving_h(node, a, b), "mv_after_h": moving_h(node, b, b + 4 * 3600), "mv_before_h": moving_h(node, a - 4 * 3600, a)})
qq = pd.concat([qq.reset_index(drop=True), pd.DataFrame(rows)], axis=1)
print(f"\nRiding hours: between tests mean {qq.mv_between_h.mean():.1f} h; in 4h after 2nd test mean {qq.mv_after_h.mean():.2f} h")
print(f"  corr(placebo air, riding after 2nd test) within region-day: "
      f"{smf.ols('pm_after_second ~ C(region_day)', qq).fit().resid.corr(smf.ols('mv_after_h ~ C(region_day)', qq).fit().resid):.2f}")
fa = base.format(x="pm_between + pm_after_second + pm_before_first") + " + C(region_day)"
show("joint + region-day FE + riding-time controls: placebo term", fa + " + mv_between_h + mv_after_h + mv_before_h", qq, "pm_after_second")
show("joint + region-day FE + riding-time controls: between-tests term", fa + " + mv_between_h + mv_after_h + mv_before_h", qq)
fit = smf.ols(fa + " + mv_between_h + mv_after_h + mv_before_h", qq).fit(cov_type="cluster", cov_kwds={"groups": qq.day_code})
for v in ("mv_between_h", "mv_after_h", "mv_before_h"):
    print(f"  {v}: {fit.params[v]:+.4f} Hz per riding-hour  p={fit.pvalues[v]:.3f}")
