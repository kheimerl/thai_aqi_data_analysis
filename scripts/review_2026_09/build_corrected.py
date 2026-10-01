"""Rebuild the PVT+AQ linked dataset with correct (Asia/Bangkok) timing, per
REANALYSIS_SPEC_2026-09-28.md. Reuses the repo's roster/PVT loaders and per-node
array cache unchanged; only the time handling differs. Also reproduces the
original (buggy) windows as a harness check against the pre-fix linked file
kept in data/superseded_buggy_tz/pvt_aq_linked.csv. Corrected sameday/1h/4h
columns match the fixed scripts/build_pvt_aq_dataset.py exactly; this script
adds the post-test placebo windows and the GPS-movement-onset window.

Output: data/review_2026_09/linked_corrected.csv
"""
import _paths  # side effect: chdir into data/review_2026_09/
import ast
import datetime as dt
import os
import warnings
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from _paths import ROOT as REPO
import build_pvt_aq_dataset as B  # noqa: E402

warnings.filterwarnings("ignore", message="Mean of empty slice")
warnings.filterwarnings("ignore", message="All-NaN slice encountered")

BKK = ZoneInfo("Asia/Bangkok")
LA = ZoneInfo("America/Los_Angeles")  # the host tz the original run silently used
POST_MIDNIGHT_CUTOFF_H = 4


def bkk_midnight(d):
    return dt.datetime.combine(d, dt.time.min, tzinfo=BKK).timestamp()


def movement_onsets():
    m = pd.read_csv("aq_minute.csv", usecols=["node", "minute", "disp_m"])
    m["moving"] = m.disp_m.between(100, 2000).astype(int)
    t = pd.to_datetime(m.minute, unit="s", utc=True).dt.tz_convert("Asia/Bangkok")
    m["date"] = t.dt.date
    m["hr"] = t.dt.hour + t.dt.minute / 60
    out = {}
    for (node, d), g in m.groupby(["node", "date"]):
        g = g.sort_values("minute")
        # >=3 moving minutes in the window starting at this minute (10 min)
        fwd = g.moving[::-1].rolling(10, min_periods=1).sum()[::-1]
        cand = g[(g.hr >= POST_MIDNIGHT_CUTOFF_H) & (g.moving == 1) & (fwd >= 3)]
        if len(cand):
            out[(node, d)] = int(cand.minute.iloc[0])
    return out


def stats_between(cache, node, lo_ts, hi_ts, include_lo=True):
    arrs = cache.get(node)
    if arrs is None:
        return dict(B.EMPTY_STATS), None
    ts = arrs["ts"]
    lo = np.searchsorted(ts, lo_ts, side="left" if include_lo else "right")
    hi = np.searchsorted(ts, hi_ts, side="right")
    return B._slice_stats(arrs, lo, hi), (ts[lo] if hi > lo else None)


def clean_speed(s, min_rt=100):
    vals = [v for v in ast.literal_eval(s) if v >= min_rt]
    return np.nan if not vals else float(np.mean([1000.0 / v for v in vals]))


def main():
    roster = B.load_roster()
    region = (pd.read_excel(f"{REPO}/2025-2026 Rider-Sensor-Summary.xlsx", sheet_name="summary")
              .assign(u=lambda r: r["PVT user"].astype(str).str.strip())
              .drop_duplicates("u").set_index("u")["Region"])
    pvt = B.load_pvt()
    cache = B.NodeCache()
    onsets = movement_onsets()

    recs = []
    for row in pvt.itertuples(index=False):
        naive = row.timestamp.to_pydatetime()
        local = naive.replace(tzinfo=BKK)
        ts_test = local.timestamp()
        study_day = (local - dt.timedelta(hours=POST_MIDNIGHT_CUTOFF_H)).date()
        hour_sd = (ts_test - bkk_midnight(study_day)) / 3600.0
        rec = {
            "timestamp": row.timestamp, "username": row.username,
            "region": region.get(row.username, np.nan),
            "response_speed_hz": row.response_speed_hz,
            "response_speed_clean_hz": clean_speed(row.response_times),
            "session_number": row.session_number,
            "hour_of_day": hour_sd, "study_day": study_day, "ts_test": ts_test,
        }
        node = B.resolve_node(row.username, study_day, roster)
        rec["node"] = node
        if node is not None:
            # --- R0 primary: study-day midnight (Bangkok) -> test
            s, first = stats_between(cache, node, bkk_midnight(study_day), ts_test)
            for k, v in s.items():
                rec[f"{k}_sameday"] = v
            rec["shift_duration_min_sameday"] = (ts_test - first) / 60 if first is not None else np.nan
            # --- S1 trailing windows, correctly aligned
            for w, sec in (("1h", 3600), ("4h", 4 * 3600)):
                s, _ = stats_between(cache, node, ts_test - sec, ts_test)
                for k, v in s.items():
                    rec[f"{k}_{w}"] = v
            # --- N1 negative control: (test, test+4h]
            s, _ = stats_between(cache, node, ts_test, ts_test + 4 * 3600, include_lo=False)
            for k, v in s.items():
                rec[f"{k}_post4h"] = v
            # --- extra placebo windows for the specification curve: (test, test+1h], (test, end of study day]
            s, _ = stats_between(cache, node, ts_test, ts_test + 3600, include_lo=False)
            for k, v in s.items():
                rec[f"{k}_post1h"] = v
            s, _ = stats_between(cache, node, ts_test, bkk_midnight(study_day) + 86400, include_lo=False)
            for k, v in s.items():
                rec[f"{k}_postday"] = v
            # --- S2 movement-onset -> test
            on = onsets.get((node, study_day))
            if on is not None and on < ts_test:
                s, _ = stats_between(cache, node, on, ts_test)
                rec["pm25_mean_onset"] = s["pm25_mean"]
                rec["temp_mean_onset"] = s["temp_mean"]
                rec["humidity_mean_onset"] = s["humidity_mean"]
                rec["n_obs_onset"] = s["n_obs"]
                rec["onset_to_test_min"] = (ts_test - on) / 60
            # --- Harness check: reproduce the ORIGINAL (buggy) sameday window
            cal = row.timestamp.date()
            bug_end = row.timestamp.timestamp()  # pandas: naive treated as UTC
            bug_lo = dt.datetime.combine(cal, dt.time.min, tzinfo=LA).timestamp()
            bug_hi = dt.datetime.combine(cal, dt.time.max, tzinfo=LA).timestamp()
            s, first = stats_between(cache, node, bug_lo, min(bug_hi, bug_end))
            rec["bug_pm25_mean_sameday"] = s["pm25_mean"]
            rec["bug_n_obs"] = s["n_obs"]
            if first is not None:
                arrs = cache.get(node)
                ts = arrs["ts"]
                lo = np.searchsorted(ts, bug_lo, "left")
                hi = np.searchsorted(ts, min(bug_hi, bug_end), "right")
                w = ts[lo:hi]
                rec["bug_frac_pre_test"] = float(np.mean(w <= ts_test)) if len(w) else np.nan
                rec["bug_win_start_bkk"] = dt.datetime.fromtimestamp(w[0], BKK).strftime("%m-%d %H:%M")
                rec["bug_win_end_bkk"] = dt.datetime.fromtimestamp(w[-1], BKK).strftime("%m-%d %H:%M")
        recs.append(rec)

    out = pd.DataFrame.from_records(recs)
    out.to_csv("linked_corrected.csv", index=False)
    print(len(out), "sessions;", out.pm25_mean_sameday.notna().sum(), "with corrected sameday PM2.5")

    old_path = os.path.join(_paths.SUPERSEDED, "pvt_aq_linked.csv")
    if os.path.exists(old_path):
        old = pd.read_csv(old_path, usecols=["timestamp", "username", "pm25_mean_sameday"])
        assert (old.timestamp.astype(str).values == out.timestamp.astype(str).values).all()
        both = old.pm25_mean_sameday.notna() & out.bug_pm25_mean_sameday.notna()
        diff = np.abs(old.pm25_mean_sameday[both] - out.bug_pm25_mean_sameday[both]).max()
        print(f"harness: buggy windows reproduce the superseded linked file on {both.sum()}/"
              f"{old.pm25_mean_sameday.notna().sum()} sessions, max |diff| {diff:.3g}")


if __name__ == "__main__":
    main()
