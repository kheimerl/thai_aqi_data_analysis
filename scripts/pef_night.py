"""Paper 2, Addendum B (SPEC_ADDENDUM.md §12, approved 2026-11-12): night air -> morning peak flow.

Run once:  python3 -I scripts/pef_night.py --pm-col pm
Morning %PB (highest value per workday-morning, PB as §1) ~ z(PM 00:00 -> reading) + days since enrolment + clock hour.
Placebo: PM reading -> 24:00. Joint model. Decision rule as written in §12. numpy/pandas only.
"""
import argparse, json, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pef_analysis import load, select, Runner, pool_rows, tost, ROOT  # noqa: E402

MIN_NMIN = 180
SESOI_PP = 2.0


def excl0(r):
    return r is not None and (r["lo_hk"] > 0 or r["hi_hk"] < 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pm-col", default="pm")
    ap.add_argument("--out", default="peakflow_paper/run_night")
    a = ap.parse_args()
    out = os.path.join(ROOT, a.out); os.makedirs(out, exist_ok=True)
    d, _ = load(a.pm_col)
    s, _ = select(d, "highest")
    mo = s[s.period == "morning"].copy()
    nc, ac = f"{a.pm_col}_night00", f"{a.pm_col}_after24"
    mo["PM_night"] = mo[nc].where(mo.pm_night00_nmin >= MIN_NMIN)
    mo["PM_after24"] = mo[ac].where(mo.pm_after24_nmin >= MIN_NMIN)
    smp = mo.dropna(subset=["PM_night", "PM_after24"])  # one shared sample for N1-N3 (spec B.0)
    C = ["days_since_enrolment", "clock_u"]
    R = Runner()
    R.run("N1", smp, "pct", [("PM_night", "z")], C, label="N1: morning %PB ~ z(night PM 00:00->reading) + days + clock")
    R.run("N2", smp, "pct", [("PM_after24", "z")], C, label="N2 placebo: z(PM reading->24:00)")
    R.run("N3", smp, "pct", [("PM_night", "z"), ("PM_after24", "z")], C, label="N3 joint")
    # EQ on N1
    t1 = R.persons[0]
    sdpct = smp.groupby("pid").pct.std()
    sd_med = float(sdpct.median())
    tsd = t1.assign(b0=t1.b0 / t1.sd_y, se0=t1.se0 / t1.sd_y)
    R.models["N1_sdunits"] = {"all:PM_night:z": pool_rows(tsd), "slot20:PM_night:z": pool_rows(tsd[tsd.slot20])}
    eq = {}
    for sset in ["all", "slot20"]:
        r = R.models["N1"][f"{sset}:PM_night:z"]
        eq[sset] = dict(pp2=tost(r, SESOI_PP), pp_02sd=tost(r, 0.2 * sd_med),
                        sd02=tost(R.models["N1_sdunits"][f"{sset}:PM_night:z"], 0.2),
                        smallest_margin=max(abs(r["lo90_hk"]), abs(r["hi90_hk"])))
    # sensitivity rows (N1)
    R.run("N1_month", smp, "pct", [("PM_night", "z")], C, dummies="month", label="N1 + calendar-month dummies")
    before = smp[~(smp.shift_start_local.notna() & (smp.timestamp_local >= smp.shift_start_local))]
    R.run("N1_beforeshift", before, "pct", [("PM_night", "z")], C, label="N1 without mornings at/after GPS shift start")
    # decision rule (spec §12 B.1), 22-person primary
    n1, n2 = R.models["N1"]["all:PM_night:z"], R.models["N2"]["all:PM_after24:z"]
    n3 = R.models["N3"]["all:PM_night:z"]
    c1 = excl0(n1) and n1["mu"] < 0
    c2 = not excl0(n2)
    c3 = excl0(n3) and n3["mu"] < 0
    rule = dict(c1_N1_excludes0_neg=bool(c1), c2_placebo_includes0=bool(c2), c3_joint_keeps_night=bool(c3),
                verdict="night-specific association" if (c1 and c2 and c3) else "not supported")
    desc = dict(mornings_selected=int(len(mo)), sample=int(len(smp)), riders=int(smp.pid.nunique()),
                before_shift_sample=int(len(before)),
                night_nmin_median=float(smp.pm_night00_nmin.median()), after24_nmin_median=float(smp.pm_after24_nmin.median()),
                within_sd_pct_mo_median=sd_med,
                within_sd_night_raw_median=float(smp.groupby("pid").PM_night.std().median()),
                corr_night_after24_within=float(smp.groupby("pid").apply(lambda g: g.PM_night.corr(g.PM_after24)).median()),
                night22_row="not run: 16 readings from 6 riders (spec §12 B.2 row 4)",
                stationary_share_row="not run: no stationary-minute column for the night window in the file")
    pd.concat(R.persons, ignore_index=True).to_csv(os.path.join(out, "per_person.csv"), index=False)
    with open(os.path.join(out, "night.json"), "w") as f:
        json.dump(dict(models=R.models, eq=eq, rule=rule, desc=desc), f, indent=1, default=float)
    print("written", out)


if __name__ == "__main__":
    main()
