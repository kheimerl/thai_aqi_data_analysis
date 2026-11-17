"""Paper 2, Addendum C (SPEC_ADDENDUM.md §13, 2026-11-12): pooled evening - morning drop. Exploratory, no PM term.

Run once:  python3 -I scripts/pef_drop.py [--timing gps|s1]
--timing gps (default until S1's columns exist): morning before GPS shift start, evening after last_move_local.
--timing s1: S1's protocol-slot labels (column names passed with --slot-col, --before, --after-home).
Writes peakflow_paper/run_drop/drop.json and per-rider CSV. numpy/pandas only.
"""
import argparse, json, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pef_stats import pool  # noqa: E402
from pef_analysis import load, select, ROOT  # noqa: E402

MIN_PAIRS = 12


def pairs(s):
    ev = s[s.period == "evening"]; mo = s[s.period == "morning"]
    keep = ["pid", "workday", "pct", "peak_flow", "timestamp_local", "shift_start_local", "last_move_local", "city_c", "slot20"]
    extra = [c for c in s.columns if c.startswith("protocol") or c in ("home_return_local", "home_return_censored_short")]
    m = mo[keep + extra].merge(ev[keep + extra], on=["pid", "workday", "city_c", "slot20"], suffixes=("_mo", "_ev"))
    m["d_pp"] = m.pct_ev - m.pct_mo
    m["d_lmin"] = m.peak_flow_ev - m.peak_flow_mo
    return m


def per_rider(m, col):
    rows = []
    for pid, g in m.groupby("pid"):
        if len(g) < MIN_PAIRS:
            continue
        rows.append(dict(pid=pid, city=g.city_c.iloc[0], slot20=bool(g.slot20.iloc[0]), n=len(g),
                         mean=g[col].mean(), se=g[col].std(ddof=1) / np.sqrt(len(g))))
    return pd.DataFrame(rows)


def pooled(t):
    if len(t) < 2:
        return None
    r = pool(t["mean"], t["se"]); r = {k: float(v) for k, v in r.items()}
    r.update(n_pairs=int(t.n.sum()), n_neg=int((t["mean"] < 0).sum()))
    return r


def block(m, col):
    t = per_rider(m, col)
    return t, dict(all=pooled(t), slot20=pooled(t[t.slot20]), BKK=pooled(t[t.city == "BKK"]), CMI=pooled(t[t.city == "CMI"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timing", default="gps", choices=["gps", "s1"])
    ap.add_argument("--slot-col", default=None); ap.add_argument("--before", default=None); ap.add_argument("--after-home", default=None)
    ap.add_argument("--drop-censored", action="store_true", help="also report C1-t without evenings whose home return is censored short (S1)")
    ap.add_argument("--out", default="peakflow_paper/run_drop")
    a = ap.parse_args()
    out = os.path.join(ROOT, a.out); os.makedirs(out, exist_ok=True)
    d, _ = load("pm")
    s, _ = select(d, "highest")
    m = pairs(s)
    res = dict(n_pairs=int(len(m)), riders=int(m.pid.nunique()), min_pairs=MIN_PAIRS,
               mean_raw=float(m.d_pp.mean()), median_raw=float(m.d_pp.median()))
    t, res["C1_pp"] = block(m, "d_pp")
    t.to_csv(os.path.join(out, "per_rider_pp.csv"), index=False)
    _, res["C1_lmin"] = block(m, "d_lmin")
    if a.timing == "gps":
        ok = (m.shift_start_local_mo.notna() & (m.timestamp_local_mo < m.shift_start_local_mo) &
              m.last_move_local_ev.notna() & (m.timestamp_local_ev >= m.last_move_local_ev))
        label = "GPS: morning before GPS shift start, evening at/after last_move_local"
    else:
        ok = (m[a.slot_col + "_mo"] == a.before) & (m[a.slot_col + "_ev"] == a.after_home)
        label = f"S1 slots: {a.slot_col} morning == {a.before}, evening == {a.after_home}"
    mt = m[ok]
    _, res["C1t_pp"] = block(mt, "d_pp")
    res["C1t_definition"] = label
    res["C1t_pairs"] = int(len(mt)); res["C1t_riders_any"] = int(mt.pid.nunique())
    if a.drop_censored:
        mc = mt[~mt.home_return_censored_short_ev.astype(str).str.lower().eq("true")]
        _, res["C1t_nocensored_pp"] = block(mc, "d_pp")
        res["C1t_nocensored_pairs"] = int(len(mc))
    with open(os.path.join(out, f"drop_{a.timing}.json"), "w") as f:
        json.dump(res, f, indent=1, default=float)
    print("written", out)


if __name__ == "__main__":
    main()
