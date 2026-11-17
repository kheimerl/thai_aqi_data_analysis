#!/usr/bin/env python3
"""Paper 2, step 0: does foundation/linked_v4_pef.csv reproduce analysis_v3 Y5a?

v3 Y5a (analysis_v3/results_v3.md Table 4): last evening reading per workday, %PB,
per-rider OLS on same-day PM + temp + rh + study day (HC1, >= 12 readings),
DL + HKSJ pooling, beta per +10 raw units. Reported: 20 riders / 503 readings,
beta -0.1245, HKSJ [-0.2583, +0.0093], p 0.066.

Differences that cannot be removed with the v4 file:
  * v3's window starts at local midnight of the workday; v4 pm_sameday starts at 04:00.
  * v3 needed >= 60 PM seconds; v4 coverage is in minutes (pm_sameday_nmin).
The onboard rh/temp columns are used here ONLY to reproduce v3; they are not ambient.

Usage: python3 -I scripts/pef_reproduce_v3.py [--pm-col pm] [--out peakflow_paper/v3_reproduction.md]
"""
import argparse, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pef_stats import rider_fit, pool, bayes

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run(df, y, x, ctrl, scale, min_n=12, label=""):
    res = [f for _, g in df.groupby("pid") if (f := rider_fit(g, y, x, ctrl, None, min_n))]
    b = np.array([r["beta"] for r in res]); s = np.array([r["se"] for r in res])
    pl = pool(b, s)
    bz = np.array([r["beta"] / r["sd_y"] for r in res]) * 10; sz = np.array([r["se"] / r["sd_y"] for r in res]) * 10
    ph = bayes(bz, sz)["p_harm"]
    n = sum(r["n"] for r in res)
    return (f"| {label} | {pl['k']} / {n} | {pl['mu']*scale:+.4f} | [{pl['lo_hk']*scale:+.4f}, {pl['hi_hk']*scale:+.4f}] "
            f"| {pl['p_hk']:.3f} | {pl['p_dl']:.3f} | {pl['i2']:.0f} | {pl['share_neg']*100:.0f}% | {ph:.2f} |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pm-col", default="pm", help="prefix of the PM columns (pm -> pm_sameday)")
    ap.add_argument("--out", default=os.path.join(ROOT, "peakflow_paper", "v3_reproduction.md"))
    a = ap.parse_args()
    d = pd.read_csv(os.path.join(ROOT, "foundation", "linked_v4_pef.csv"))
    d = d[d.role == "rider"].copy()
    t = pd.to_datetime(d.timestamp_local)
    unix_check = ((t - pd.Timestamp("1970-01-01")).dt.total_seconds() - 7 * 3600 - d.reading_unix).abs().max()
    d["naive"] = t
    d = d[~d.exact_duplicate].copy()
    d = d[d.peak_flow >= 50].copy()          # v3 dropped < 50 L/min (= F1)
    h = d.naive.dt.hour
    d["slot"] = np.where((h >= 4) & (h < 13), "morning", np.where((h >= 15) | (h < 4), "evening", "other"))
    d["studyday"] = (d.naive - d.groupby("pid").naive.transform("min").dt.normalize()).dt.days
    ev = d[d.slot == "evening"].sort_values("naive").groupby(["pid", "workday"]).last().reset_index()
    pmc = f"{a.pm_col}_sameday"
    ev["pm_v3"] = np.where(ev.pm_sameday_nmin >= 1, ev[pmc], np.nan)
    hdr = ("| Model | riders / n | β per +10 raw units (HKSJ) | 95% CI (HKSJ) | p HKSJ | p DL | I² % | riders β<0 | P(harm) |\n"
           "|---|---|---|---|---|---|---|---|---|")
    L = ["# v3 Y5a reproduction from linked_v4_pef.csv (S4, step 0)", "",
         f"Clock cross-check: max |reading_unix − (timestamp_local − 7 h)| = {unix_check:.0f} s over all {len(t)} rows.", "",
         f"Rider rows after dropping the exact duplicate and < 50 L/min: {len(d)}; evening workdays (v3 slot rule): {len(ev)}; "
         f"with same-day PM: {ev.pm_v3.notna().sum()}.", "",
         "| Source | riders / n | β per +10 | 95% CI (HKSJ) | p HKSJ | p DL | I² % | riders β<0 | P(harm) |", "|---|---|---|---|---|---|---|---|---|",
         "| **v3 as reported** (midnight → reading, temp + rh + study day) | 20 / 503 | −0.1245 | [−0.2583, +0.0093] | 0.066 | 0.042 | 38 | 70% | 0.98 |", "",
         "## v4 file, v3 model", hdr]
    L.append(run(ev, "pct_pb", "pm_v3", ["temp_sameday", "rh_sameday", "studyday"], 10, label="v3 model on v4 (04:00 → reading)"))
    ev2 = ev[ev.pm_sameday_nmin >= 60]
    L.append(run(ev2, "pct_pb", "pm_v3", ["temp_sameday", "rh_sameday", "studyday"], 10, label="same, same-day ≥ 60 min of PM"))
    ev3 = ev[~ev.F5_outside_enrolment]
    L.append(run(ev3, "pct_pb", "pm_v3", ["temp_sameday", "rh_sameday", "studyday"], 10, label="same, F5 dropped"))
    # use S1's own last-evening flag instead of the v3 slot rule
    ev4 = d[d.last_evening_of_workday].copy(); ev4["pm_v3"] = ev4[pmc]
    L.append(run(ev4, "pct_pb", "pm_v3", ["temp_sameday", "rh_sameday", "studyday"], 10, label="S1 last_evening_of_workday flag"))
    L.append(run(ev, "pct_pb", "pm_v3", ["studyday"], 10, label="v3 model without onboard temp/rh"))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    open(a.out, "w").write("\n".join(L) + "\n"); print("\n".join(L))


if __name__ == "__main__":
    main()
