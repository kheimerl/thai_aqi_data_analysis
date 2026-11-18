"""Paper 2, Addendum E (SPEC_ADDENDUM.md §15, 2026-11-17). Exploratory.
Idea 2: personal-above-city PM, rider FE + city x workday FE. Idea 3: haze-onset difference-in-differences.
python3 -I scripts/pef_added.py --pm-col pm  -> peakflow_paper/run_added/added.json. numpy/pandas only.
"""
import argparse, json, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pef_analysis import load, select, ROOT  # noqa: E402
from pef_models import crfit  # noqa: E402

ONSET = pd.Timestamp("2026-03-06")  # SPEC §15, fixed from PM only
DROP = ["R09", "R14", "R20"]


def twfe(df, y, xcols, ctrls, g2):
    """Rider FE + g2 FE by alternating demeaning; CR1 by rider. Returns coef of xcols[0]."""
    cols = [y] + xcols + ctrls
    M = df[cols].astype(float).copy()
    for _ in range(200):
        old = M.copy()
        M = M - M.groupby(df.pid).transform("mean")
        M = M - M.groupby(df[g2]).transform("mean")
        if (M - old).abs().to_numpy().max() < 1e-10:
            break
    X = M[xcols + ctrls].to_numpy(); names = xcols + [None] * len(ctrls)
    r, info = crfit(M[y].to_numpy(), X, df.pid.to_numpy(), names, absorbed=df.pid.nunique() + df[g2].nunique())
    return dict(**r[xcols[0]], **info, groups2=int(df[g2].nunique()), riders=int(df.pid.nunique()))


def idea2(rows, y, xcol, ctrls, drop=None):
    d = rows.dropna(subset=[y, xcol] + ctrls).copy()
    d = d[d[xcol] > 0]
    if drop:
        d = d[~d.pid.isin(drop)]
    d["cd"] = d.city_c + "_" + d.workday.astype(str)
    d = d[d.groupby("cd").pid.transform("nunique") >= 2]
    d = d[d.groupby("pid")[y].transform("size") >= 2]
    lx = np.log(d[xcol]); sdw = float(lx.groupby(d.pid).std().median())
    d["x"] = lx / sdw
    r = twfe(d, y, ["x"], ctrls, "cd"); r["sd_log"] = sdw
    return r


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--pm-col", default="pm"); a = ap.parse_args()
    out = os.path.join(ROOT, "peakflow_paper/run_added"); os.makedirs(out, exist_ok=True)
    d, _ = load(a.pm_col); s, _ = select(d, "highest")
    s["wd"] = pd.to_datetime(s.workday)
    ev = s[s.period == "evening"].copy(); mo = s[s.period == "morning"].copy()
    mo["PM_night"] = mo[f"{a.pm_col}_night00"].where(mo.pm_night00_nmin >= 180)
    mo["PM_after24"] = mo[f"{a.pm_col}_after24"].where(mo.pm_after24_nmin >= 180)
    CE, CM = ["clock_u", "hours_on_shift_before_test"], ["clock_u"]
    R = {}
    R["E2a_evening"] = idea2(ev, "pct", "PM_sd", CE)
    R["E2a_placebo_after"] = idea2(ev, "pct", "PM_after", CE)
    R["E2a_no_R09_R14_R20"] = idea2(ev, "pct", "PM_sd", CE, DROP)
    R["E2b_morning_night"] = idea2(mo, "pct", "PM_night", CM)
    R["E2b_placebo_after24"] = idea2(mo, "pct", "PM_after24", CM)
    R["E2b_no_R09_R14_R20"] = idea2(mo, "pct", "PM_night", CM, DROP)
    # Idea 3
    e3 = pd.read_csv(os.path.join(ROOT, "calibration/tables/e3_city_daily.csv"), parse_dates=["date"]).set_index("date")
    for half in (14, 21):
        lo, hi = ONSET - pd.Timedelta(days=half), ONSET + pd.Timedelta(days=half)
        w = e3[(e3.index >= lo) & (e3.index <= hi)]; pre = w.index < ONSET
        pm = {c: float(w.loc[~pre, f"{c.lower()}_rider_median_raw"].mean() / w.loc[pre, f"{c.lower()}_rider_median_raw"].mean()) for c in ("CMI", "BKK")}
        for per, rows in (("morning", mo), ("evening", ev)):
            x = rows[(rows.wd >= lo) & (rows.wd <= hi)].dropna(subset=["pct"]).copy()
            x["post"] = (x.wd >= ONSET).astype(float); x["did"] = x.post * (x.city_c == "CMI")
            x = x[x.groupby("pid").pct.transform("size") >= 2]
            r = twfe(x, "pct", ["did"], [], "workday")
            rm = x.groupby(["city_c", "pid", "post"]).pct.mean().unstack()
            raw = {c: dict(before=float(rm.loc[c][0.0].mean()), after=float(rm.loc[c][1.0].mean()),
                           change_riders_both=float((rm.loc[c][1.0] - rm.loc[c][0.0]).mean()),
                           riders_both=int(rm.loc[c].dropna().shape[0])) for c in ("CMI", "BKK")}
            R[f"E3_{per}_pm{half}"] = dict(**r, dates=int(x.workday.nunique()), raw=raw, pm_ratio_after_before=pm,
                                           dates_pre=int((x.wd < ONSET).groupby(x.workday).first().sum()))
    json.dump(R, open(os.path.join(out, "added.json"), "w"), indent=1)
    for k, v in R.items():
        print(f"{k:22s} {v['b']:+.2f} [{v['lo']:+.2f}, {v['hi']:+.2f}] p {v['p']:.3f} n {v['n']} riders {v['riders']} groups {v['groups2']}"
              + (f" | raw {json.dumps(v['raw'])} pm {v['pm_ratio_after_before']} dates_pre {v['dates_pre']}" if 'raw' in v else ""))


if __name__ == "__main__":
    main()
