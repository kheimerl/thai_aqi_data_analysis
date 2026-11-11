"""S1 Task 1: independent rebuild of the PVT x same-day PM2.5 primary model (Gate 0).

Written without reading analysis_health_v3.py / analysis_v3 / exploratory_spike_tz.
Inputs: data/pvt.csv, mapping, roster xlsx, foundation/_work/aq_min/<node>.csv
Rules chosen here (stated in REPRODUCTION.md):
  * riders = mapping 'PVT user' (usernames stripped of whitespace)
  * person->sensor from roster xlsx (Start..End inclusive, Bangkok dates)
  * session speed = mean(1000/RT) over response_times with RT >= 100 ms
  * PM = second-weighted mean of valid PM (0..500) on that node from local midnight
    of the test day (first data = sensor-on) up to the test minute
  * per-rider OLS speed ~ 1 + PM10 + day_of_study (days since roster start), riders with
    >= 10 PM-covered sessions and PM variance > 0; DL + HKSJ pooling
  * clock: test_unix = local - 7 h  (variant 'LA' emulates a Seattle machine)
"""
import os, sys, ast, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from foundation_common import *

W = os.path.join(FOUND, "_work"); AQM = os.path.join(W, "aq_min")


def roster():
    x = read_xlsx(os.path.join(DATA, "2025-2026 Rider-Sensor-Summary.xlsx"))["summary"]
    x = x.iloc[1:]
    r = pd.DataFrame({"user": x["D"].astype(str).str.strip(), "sensor": x["E"].astype(float).astype(int),
                      "start": x["F"].map(excel_serial_to_date), "end": x["G"].map(excel_serial_to_date),
                      "region": x["A"].str.strip()})
    return r.dropna(subset=["start"])


class NodePM:
    def __init__(self):
        self.c = {}

    def get(self, node):
        if node not in self.c:
            f = os.path.join(AQM, f"{node}.csv")
            if not os.path.exists(f):
                self.c[node] = None
            else:
                d = pd.read_csv(f)
                self.c[node] = (d.minute_unix.values, np.concatenate([[0], np.cumsum(d.pm_sum.values)]),
                                np.concatenate([[0], np.cumsum(d.n_pm.values)]),
                                np.concatenate([[0], np.cumsum((d.n_pm.values > 0).astype(int))]))
        return self.c[node]

    def window(self, node, u0, u1):
        """mean PM, n seconds, n minutes for minute_unix in [u0, u1)."""
        a = self.get(node)
        if a is None: return np.nan, 0, 0
        mu, cs, cn, cm = a
        i, j = np.searchsorted(mu, u0, "left"), np.searchsorted(mu, u1, "left")
        n = cn[j] - cn[i]
        return ((cs[j] - cs[i]) / n if n > 0 else np.nan), int(n), int(cm[j] - cm[i])


def session_speed(rts, lo=100):
    rt = np.asarray(rts, float); rt = rt[rt >= lo]
    return np.mean(1000 / rt) if len(rt) else np.nan, len(rt)


def build(clock="BKK"):
    m = read_mapping(); ro = roster()
    p = pd.read_csv(os.path.join(DATA, "pvt.csv"))
    p["user"] = p.username.str.strip()
    p = p[p.user.isin(set(m.user))].copy()
    p["local"] = pd.to_datetime(p.timestamp)
    if clock == "BKK":
        p["test_unix"] = local_to_unix(p["local"])
    else:  # emulate datetime.timestamp() on a machine set to America/Los_Angeles
        loc = p["local"].dt.tz_localize("America/Los_Angeles", ambiguous="NaT", nonexistent="shift_forward")
        p["test_unix"] = (loc.dt.tz_convert("UTC").dt.tz_localize(None) - EPOCH).dt.total_seconds()
    p["date"] = p["local"].dt.normalize()
    sp = p.response_times.map(lambda s: session_speed(ast.literal_eval(s)))
    p["speed"] = sp.map(lambda t: t[0]); p["n_trials"] = sp.map(lambda t: t[1])
    # person -> sensor by roster date range
    p["sensor"] = np.nan; p["start"] = pd.NaT
    for _, r in ro.iterrows():
        sel = (p.user == r.user) & (p.date >= r.start) & (p.date <= r.end)
        p.loc[sel, "sensor"] = r.sensor
    st = ro.groupby("user").start.min(); p["start"] = p.user.map(st)
    p["day_of_study"] = (p.date - p.start).dt.days
    p["city"] = p.user.map(m.set_index("user").city)
    npm = NodePM(); res = []
    for _, r in p.iterrows():
        if np.isnan(r.sensor):
            res.append((np.nan, 0, 0)); continue
        # day boundary: Bangkok local midnight of the test date, in Unix
        d0 = local_to_unix(r.date)
        if clock == "BKK":
            u0 = d0
        else:  # the buggy build would also place the day boundary on the LA clock
            u0 = r.test_unix - (r.local - r.date).total_seconds()
        res.append(npm.window(int(r.sensor), u0, r.test_unix))
    p["pm_sameday"] = [x[0] for x in res]; p["pm_n_sec"] = [x[1] for x in res]; p["pm_n_min"] = [x[2] for x in res]
    return p


def fit(p, min_sessions=10, col="pm_sameday", pm_min_minutes=1):
    d = p[p[col].notna() & p.speed.notna() & (p.pm_n_min >= pm_min_minutes)]
    rows = []
    for u, g in d.groupby("user"):
        if len(g) < min_sessions or g[col].std() == 0: continue
        X = np.column_stack([np.ones(len(g)), g[col] / 10, g.day_of_study])
        b, se = ols(g.speed, X)
        rows.append(dict(user=u, n=len(g), b=b[1], se=se[1]))
    per = pd.DataFrame(rows)
    pool = pool_dl_hksj(per.b, per.se)
    pool["n_sessions"] = int(per.n.sum()); pool["n_used_available"] = int(len(d))
    return per, pool


if __name__ == "__main__":
    out = {}
    for clock in ["BKK", "LA"]:
        p = build(clock)
        p.to_csv(os.path.join(W, f"task1_sessions_{clock}.csv"), index=False)
        per, pool = fit(p)
        per.to_csv(os.path.join(W, f"task1_perrider_{clock}.csv"), index=False)
        out[clock] = dict(total_sessions=len(p), with_sensor=int(p.sensor.notna().sum()),
                          with_pm=int(p.pm_sameday.notna().sum()), **pool)
        if clock == "BKK":
            for mm in [30, 60]:
                _, pl = fit(p, pm_min_minutes=mm); out[f"BKK_min{mm}min"] = pl
    print(json.dumps(out, indent=1, default=str))
