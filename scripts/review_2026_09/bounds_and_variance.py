"""POST-HOC summary numbers cited in the bounded-null memo:

1. Interval bounds converted to ms of mean-RT change per +50 ug/m3.
2. Between-day variance of daily PM2.5 by region, and its rider/day split.
3. Share of measured daily PM2.5 dose (concentration x time) at home vs on shift.
4. Cross-rider dependence of daily exposure and of per-rider residuals (Pesaran CD).

Needs linked_corrected.csv (build_corrected.py) and minute_classes.csv (home_road.py).
"""
import _paths  # side effect: chdir into data/review_2026_09/
import itertools
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")

# Estimates (x1e-4 Hz per ug/m3) and 95% CIs as printed by the scripts named.
ESTIMATES = [
    ("Same city, same day FE (run_reanalysis N2b, declared)", -0.54, -3.70, 2.61),
    ("Plan of record, timing fixed (run_reanalysis R1, declared)", -2.05, -4.71, 0.61),
    ("Consecutive study days (adjacent_days A, post-hoc)", -1.37, -6.07, 3.32),
    ("Same day + rider-week FE (remove_drift E, post-hoc)", -4.0, -10.2, 2.3),
    ("Within-day change + same-day FE (within_day, post-hoc)", -5.77, -14.32, 2.78),
    ("Original result, timing bug (superseded)", -6.41, -12.54, -0.29),
]


def main():
    c = pd.read_csv("linked_corrected.csv").dropna(subset=["response_speed_hz", "pm25_mean_sameday"])
    S = c.response_speed_hz.mean()
    wsd = (c.response_speed_hz - c.groupby("username").response_speed_hz.transform("mean")).std()
    ms = lambda b: 1000 / (S + b * 1e-4 * 50) - 1000 / S
    print(f"1. Bounds per +50 ug/m3 (mean speed {S:.3f} Hz ~ {1000/S:.0f} ms; within-rider session SD {wsd:.3f} Hz)")
    for lab, b, lo, hi in ESTIMATES:
        print(f"   {lab:60s} point {ms(b):+5.1f} ms  95% [{ms(hi):+5.1f}, {ms(lo):+5.1f}] ms  worst case = {abs(lo*1e-4*50)/wsd:.2f} SD")

    m = pd.read_csv("minute_classes.csv").dropna(subset=["pm25"])
    rd = (m.groupby(["region", "rider", "date"])
          .apply(lambda g: pd.Series({"pm": np.average(g.pm25, weights=g.n), "hours": g.n.sum() / 3600})).reset_index())
    rd = rd[rd.hours >= 12]
    day = rd.groupby(["region", "date"]).pm.mean().reset_index()
    day["month"] = pd.to_datetime(day.date).dt.month
    day["week"] = pd.to_datetime(day.date).dt.isocalendar().week.astype(str)
    print("\n2. Between-day variation of the city's daily (24h) mean PM2.5, ug/m3")
    for reg, g in day.groupby("region"):
        mo, wk = smf.ols("pm ~ C(month)", g).fit(), smf.ols("pm ~ C(week)", g).fit()
        print(f"   {reg:10s} days={len(g)} mean={g.pm.mean():.1f} SD={g.pm.std():.1f} var={g.pm.var():.0f} "
              f"range {g.pm.min():.0f}-{g.pm.max():.0f}; month explains {100*mo.rsquared:.0f}%; within-week SD {np.sqrt(wk.mse_resid):.1f}")
    rd["cd"] = rd.region + "_" + rd.date
    tot = ((rd.pm - rd.pm.mean()) ** 2).sum()
    ss_cd = ((smf.ols("pm ~ C(cd)", rd).fit().fittedvalues - rd.pm.mean()) ** 2).sum()
    full = smf.ols("pm ~ C(cd) + C(rider)", rd).fit()
    print(f"   rider-day means (n={len(rd)}, SD {rd.pm.std():.1f}): city-day {100*ss_cd/tot:.0f}%, "
          f"rider {100*(full.rsquared - ss_cd/tot):.0f}%, rider x day {100*(1 - full.rsquared):.0f}% (SD {np.sqrt(full.mse_resid):.1f})")

    m["dose"] = m.pm25 * m.n
    m["mo"] = pd.to_datetime(m.date).dt.month
    def share(d):
        s = d.groupby("cls").dose.sum()
        return (100 * s / s.sum()).round(0).to_dict()
    print("\n3. Share of measured PM2.5 dose by location class (%)")
    print("   all:", share(m))
    print("   Chiang Mai Mar-Apr:", share(m[(m.region == "Chiang mai") & m.mo.isin([3, 4])]))

    cov = "temp_mean_sameday + humidity_mean_sameday + shift_duration_min_sameday + session_number + hour_of_day"
    d = c.dropna(subset=["temp_mean_sameday", "humidity_mean_sameday", "shift_duration_min_sameday"])
    parts = []
    for u, g in d.groupby("username"):
        if len(g) >= 15:
            parts.append(g.assign(e=smf.ols(f"response_speed_hz ~ pm25_mean_sameday + {cov}", g).fit().resid))
    d = pd.concat(parts)
    rdd = d.groupby(["region", "username", "study_day"]).agg(e=("e", "mean"), pm=("pm25_mean_sameday", "mean")).reset_index()
    rdd["pm"] = rdd.pm - rdd.groupby("username").pm.transform("mean")
    print("\n4. Cross-rider dependence within region (mean pairwise corr; Pesaran CD)")
    for var, lab in (("pm", "daily PM2.5 (rider-demeaned)"), ("e", "per-rider model residuals")):
        for reg, g in rdd.groupby("region"):
            w = g.pivot(index="study_day", columns="username", values=var)
            rs, ns = [], []
            for a, b in itertools.combinations(w.columns, 2):
                x = w[[a, b]].dropna()
                if len(x) >= 8:
                    rs.append(x.corr().iloc[0, 1]); ns.append(len(x))
            rs, ns = np.array(rs), np.array(ns)
            cd = np.sqrt(2 / (len(w.columns) * (len(w.columns) - 1))) * np.sum(np.sqrt(ns) * rs)
            print(f"   {lab:30s} {reg:10s} r={rs.mean():+.2f}  CD={cd:.1f}")


if __name__ == "__main__":
    main()
