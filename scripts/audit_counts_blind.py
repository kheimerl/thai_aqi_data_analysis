#!/usr/bin/env python3
"""Blind data audit for the pre-registration (analysis plan, step 2).

COUNTS ONLY. This script never loads aq_sensor.csv and never links any outcome to PM2.5,
so it is safe to run BEFORE the analysis plan is filed.

Answers:
  1. How many peak-flow readings exist (per rider, per day, per sitting)?
  2. How many same-day PVT pairs exist (and how many survive minimum-gap rules)?
  3. How many rider-days have both peak flow and PVT?

Usernames are replaced by R01, R02, ... in all output. The mapping is NOT saved.

Usage:
  python3 scripts/audit_counts_blind.py --pvt PATH/pvt.csv --peakflow PATH/peakflow.csv --out OUTDIR
Optional:
  --office name1,name2   usernames of office workers (analyzed separately, excluded from rider counts)
  --min-pef-readings 10 --min-pef-days 5   placeholder thresholds to be fixed in the registration
Assumptions to verify in the printed header:
  * numeric timestamps are Unix epoch (s or ms); text timestamps without a zone are Bangkok local time.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

TZ = "Asia/Bangkok"
# Same staff / test accounts that scripts/build_pvt_aq_dataset.py excludes.
EXCLUDE_USERNAMES = {"Nussara", "Rach", "Rac", "test", "Preechai", "adisorn", "Patte"}
SITTING_GAP_MIN = 10  # readings closer than this (same user) count as one sitting (repeat blows)
GAP_RULES_MIN = (0, 30, 60, 120)  # minimum minutes between the two tests of a pair
MIN_PVT_SESSIONS = 15  # existing rule in PLAN.md


def to_local_naive(s: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s):
        unit = "ms" if s.dropna().median() > 1e11 else "s"
        dt = pd.to_datetime(s, unit=unit, utc=True).dt.tz_convert(TZ)
        return dt.dt.tz_localize(None)
    dt = pd.to_datetime(s, errors="coerce")
    if getattr(dt.dt, "tz", None) is not None:
        dt = dt.dt.tz_convert(TZ).dt.tz_localize(None)
    return dt


def load(path, cols):
    df = pd.read_csv(path, usecols=lambda c: c in cols)
    missing = set(cols) - set(df.columns)
    if missing:
        sys.exit(f"{path}: missing columns {sorted(missing)}")
    print(f"  raw timestamp examples: {df['timestamp'].head(3).tolist()}")
    df["username"] = df["username"].astype(str).str.strip()
    df["dt"] = to_local_naive(df["timestamp"])
    n0 = len(df)
    df = df.dropna(subset=["dt"])
    df = df[~df["username"].isin(EXCLUDE_USERNAMES)]
    dups = df.duplicated(["username", "dt"]).sum()
    df = df.drop_duplicates(["username", "dt"]).sort_values(["username", "dt"])
    df["date"] = df["dt"].dt.normalize()
    print(f"  rows read {n0}, after staff/NaT filter and de-dup {len(df)} (exact duplicates removed: {dups})")
    return df


def with_sittings(pef):
    prev = pef.groupby("username")["dt"].shift()
    gap = (pef["dt"] - prev).dt.total_seconds() / 60
    pef = pef.copy()
    pef["new_sitting"] = gap.isna() | (gap > SITTING_GAP_MIN)
    return pef


def pair_table(pvt):
    p = pvt.copy()
    p["prev_dt"] = p.groupby(["username", "date"])["dt"].shift()
    p["gap_min"] = (p["dt"] - p["prev_dt"]).dt.total_seconds() / 60
    return p.dropna(subset=["gap_min"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pvt", required=True)
    ap.add_argument("--peakflow", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--office", default="")
    ap.add_argument("--min-pef-readings", type=int, default=10)
    ap.add_argument("--min-pef-days", type=int, default=5)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    office = {x.strip() for x in a.office.split(",") if x.strip()}

    print("PVT file")
    pvt = load(a.pvt, ["timestamp", "username", "valid_taps", "total_taps"])
    print("Peak-flow file")
    pef = load(a.peakflow, ["timestamp", "username", "peak_flow"])

    users = sorted(set(pvt.username) | set(pef.username))
    ids = {u: f"R{i + 1:02d}" for i, u in enumerate(users)}
    for d in (pvt, pef):
        d["rid"] = d["username"].map(ids)
        d["group"] = np.where(d["username"].isin(office), "office", "rider")

    pef = with_sittings(pef)
    pairs = pair_table(pvt)

    # ---------- peak flow ----------
    print("\n=== PEAK FLOW ===")
    print(f"date range: {pef.dt.min()} to {pef.dt.max()}")
    v = pef["peak_flow"]
    print(f"values (L/min): min {v.min()}, median {v.median()}, max {v.max()}; "
          f"<=0: {(v <= 0).sum()}, <50: {(v < 50).sum()}, >900: {(v > 900).sum()}, missing: {v.isna().sum()}")
    print("(implausible-value cutoffs above are for flagging only; fix the real rule in the registration)")
    hours = pef.groupby(pef.dt.dt.hour).size()
    print("readings by hour of day:", hours.to_dict())

    # ---------- per-rider table ----------
    rows = []
    for u in users:
        pv = pvt[pvt.username == u]
        pf = pef[pef.username == u]
        pr = pairs[pairs.username == u]
        pv_days = set(pv.date)
        pf_days = set(pf.date)
        pf_day_counts = pf.groupby("date")["new_sitting"].sum()
        rows.append({
            "rid": ids[u],
            "group": "office" if u in office else "rider",
            "pvt_sessions": len(pv),
            "pvt_days": len(pv_days),
            "pvt_days_ge2": int((pv.groupby("date").size() >= 2).sum()),
            "pvt_pairs_any_gap": len(pr),
            "pvt_pairs_gap_ge60": int((pr["gap_min"] >= 60).sum()),
            "pef_readings": len(pf),
            "pef_sittings": int(pf["new_sitting"].sum()),
            "pef_days": len(pf_days),
            "pef_days_ge2_sittings": int((pf_day_counts >= 2).sum()),
            "days_with_both": len(pv_days & pf_days),
        })
    per = pd.DataFrame(rows)
    per.to_csv(os.path.join(a.out, "per_rider_counts.csv"), index=False)

    for grp in ("rider", "office"):
        g = per[per.group == grp]
        if g.empty:
            continue
        print(f"\n=== GROUP: {grp.upper()}  (n = {len(g)}) ===")
        print(f"PVT sessions total: {g.pvt_sessions.sum()}; riders with >= {MIN_PVT_SESSIONS} sessions: "
              f"{(g.pvt_sessions >= MIN_PVT_SESSIONS).sum()}")
        print(f"Peak-flow readings total: {g.pef_readings.sum()} in {g.pef_sittings.sum()} sittings; "
              f"riders with any: {(g.pef_readings > 0).sum()}")
        ok = ((g.pef_readings >= a.min_pef_readings) & (g.pef_days >= a.min_pef_days)).sum()
        print(f"Riders with >= {a.min_pef_readings} PEF readings on >= {a.min_pef_days} days (placeholder rule): {ok}")
        print(f"Rider-days with both PEF and PVT: {g.days_with_both.sum()}")
        print(f"Rider-days with >= 2 PEF sittings: {g.pef_days_ge2_sittings.sum()}")
        print("Same-day PVT pairs (consecutive tests, same rider, same calendar day):")
        gp = pairs[pairs.group == grp]
        for m in GAP_RULES_MIN:
            sel = gp[gp.gap_min >= m]
            print(f"  gap >= {m:>3} min: pairs {len(sel):>5}, riders {sel.rid.nunique():>3}, "
                  f"rider-days {sel.groupby(['rid', 'date']).ngroups:>5}")
        if len(gp):
            print(f"  gap between tests (min): median {gp.gap_min.median():.0f}, "
                  f"IQR {gp.gap_min.quantile(.25):.0f}-{gp.gap_min.quantile(.75):.0f}, max {gp.gap_min.max():.0f}")

    # ---------- timing: PEF relative to PVT, same rider-day ----------
    print("\n=== TIMING (no exposure involved) ===")
    m = pef[["rid", "dt", "date"]].merge(pvt[["rid", "dt", "date"]], on=["rid", "date"], suffixes=("_pef", "_pvt"))
    if len(m):
        m["diff"] = (m.dt_pef - m.dt_pvt).dt.total_seconds().abs() / 60
        nearest = m.groupby(["rid", "dt_pef"])["diff"].min()
        print(f"PEF readings with a PVT the same day: {len(nearest)} of {len(pef)}; "
              f"nearest-PVT distance (min): median {nearest.median():.0f}, "
              f"share within 30 min {np.mean(nearest <= 30):.0%}")
    else:
        print("No rider-days with both PEF and PVT.")

    print(f"\nPer-rider counts written to {os.path.join(a.out, 'per_rider_counts.csv')} (pseudonymous IDs)")
    print("NOT checked here: PM2.5 coverage at the test/reading times. A pair or reading is only usable "
          "if the same-day exposure exists; count that as the next blind step (coverage, not effects).")


if __name__ == "__main__":
    main()
