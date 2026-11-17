"""Home location, home stays, shift episodes and home-return time (S1, 2026-11-12, coordinator request).

GPS only: no PM, no outcome. Rule text: foundation/QC_RULES.md change log 2026-11-12 ("home return").
Coordinates stay in foundation/_work/ (internal); shared outputs carry times and labels only.

Steps (per person-period = pid x sensor x roster window; office O01-O04 over their sensor's whole span):
  1. Night location for each calendar date d: stationary GPS fixes (speed <= 5 km/h) in 00:00-04:00 local;
     mode 0.001-degree cell, then median of fixes within 100 m of that cell's median (>= 120 such fixes, i.e.
     >= 2 h parked there). Fallback, only for a person-period with < 2 such nights (GPS gets no fix indoors
     overnight): the same mode location from stationary fixes in 22:00-24:00 of d (>= 30 fixes), on dates without
     an overnight location; the period's overnight location, if it has one, is kept too. (The export holds isolated study days 00:00-24:00,
     so the previous evening is usually not in the data.)
  2. Homes: greedy clustering of night locations (CLUSTER_M); every cluster with >= 2 nights is a home.
     A person can have several homes (moved house, a second place); single nights elsewhere are not homes.
  3. Minute state: H = fix within RADIUS of any home, A = fix elsewhere (only runs of >= 3 consecutive fixes;
     shorter runs are GPS jumps), N = sensor on, no fix.
     Home stay: starts at the first H minute, continues over H and N minutes and over sensor-off gaps <= GAP_MAX,
     ends at the first A minute (departure). A gap > GAP_MAX or the end of data ends it as censored.
     Long stay = observed duration >= MIN_STAY, or censored at the data end after >= CENS_MIN minutes.
  4. Away period = time between two long stays, split at data gaps > GAP_MAX. A shift episode = such a block
     with >= 1 sustained-movement minute (QC_RULES D: moving and >= 8 of the 15 minutes before or after moving).
     home_return = start of the next long stay if the block runs straight into it; NA if the data end first.
  5. Workday (04:00-04:00, unchanged) -> episodes whose first sustained movement falls in it; day's home_return
     = return of the last of them (may fall after 04:00). Crosses 04:00 = that episode has sustained movement in
     the 2 h before the next 04:00 and sustained movement or the home return in the 2 h after it.
"""
import os, sys, argparse
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from foundation_common import FOUND, local_to_unix, unix_to_local

W = os.path.join(FOUND, "_work")
RADIUS = 150        # m, at-home radius (chosen from overnight jitter, see QC_RULES)
CLUSTER_M = 200     # m, night-location clustering
MIN_STAY = 3 * 3600 # s
CENS_MIN = 15 * 60  # s
GAP_MAX = 12 * 3600 # s
A_RUN = 3


def hav(la1, lo1, la2, lo2):
    la1, lo1, la2, lo2 = map(np.radians, (la1, lo1, la2, lo2))
    a = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return 2 * 6371000 * np.arcsin(np.sqrt(a))


def periods():
    from foundation_ids import assignments
    a = assignments().copy(); a["role"] = "rider"
    C = pd.read_csv(os.path.join(FOUND, "cohort.csv"))
    o = C[(C.role != "rider") & (C.pid != "O05")][["pid", "sensor"]].copy()
    o["start"] = pd.Timestamp("2025-12-01"); o["end"] = pd.Timestamp("2026-05-01"); o["role"] = "office"
    o["sensor"] = o.sensor.astype(int)
    return pd.concat([a, o], ignore_index=True)


MIN = {}
def load(node):
    if node not in MIN:
        d = pd.read_csv(os.path.join(W, f"minute_{node}.csv"), usecols=["minute_unix", "n_gps", "lat", "lon", "speed_kmh"])
        d = d.sort_values("minute_unix").reset_index(drop=True)
        loc = pd.Series(unix_to_local(d.minute_unix), index=d.index)
        d["date"] = loc.dt.normalize(); d["hour"] = loc.dt.hour
        d["fix"] = (d.n_gps > 0) & d.lat.notna()
        d["moving"] = d.speed_kmh > 5
        u = d.minute_unix.values; mv = d.moving.values
        # sustained movement on the continuous minute grid (same counts as QC_RULES D)
        idx = ((u - u[0]) // 60).astype(int); grid = np.zeros(idx[-1] + 16, bool); grid[idx] = mv
        cs = np.concatenate([[0], np.cumsum(grid)])
        fwd = cs[idx + 15] - cs[idx]
        back = cs[idx + 1] - cs[np.maximum(0, idx - 14)]
        d["sustained"] = mv & ((fwd >= 8) | (back >= 8))
        MIN[node] = d
    return MIN[node]


def mode_loc(g):
    cell = (g.lat.round(3).astype(str) + "_" + g.lon.round(3).astype(str))
    top = cell.value_counts().index[0]; s = g[cell == top]
    la, lo = s.lat.median(), s.lon.median()
    k = g[hav(g.lat, g.lon, la, lo) < 100]
    return k.lat.median(), k.lon.median(), len(k)


def night_locations(p):
    d = load(p.sensor)
    d = d[(d.date >= p.start) & (d.date <= p.end + pd.Timedelta(days=1))]
    out = []
    for date, g in d.groupby("date"):
        n = g[(g.hour < 4) & g.fix & ~g.moving]
        if len(n) >= 120:
            la, lo, k = mode_loc(n)
            if k >= 120:
                out.append(dict(pid=p.pid, sensor=p.sensor, date=date, lat=la, lon=lo, n_fix=k, source="night_fix")); continue
        ev = g[(g.hour >= 22) & g.fix & ~g.moving]
        if len(ev) >= 30:
            la, lo, k = mode_loc(ev)
            if k >= 30:
                out.append(dict(pid=p.pid, sensor=p.sensor, date=date, lat=la, lon=lo, n_fix=k, source="evening_fix"))
    out = pd.DataFrame(out)
    # the 22:00-24:00 fallback is used only for a person-period whose GPS has < 2 overnight (00-04) locations;
    # then any overnight location is kept alongside the evening ones (S7 SF2, 2026-11-12; no period affected)
    if len(out) and (out.source == "night_fix").sum() >= 2:
        out = out[out.source == "night_fix"]
    return out.reset_index(drop=True)


def cluster(N):
    N = N.copy().reset_index(drop=True); N["cluster"] = -1; c = 0
    while (N.cluster == -1).any():
        r = N[N.cluster == -1]
        cnt = [(hav(r.lat, r.lon, a, b) <= CLUSTER_M).sum() for a, b in zip(r.lat, r.lon)]
        i = r.index[int(np.argmax(cnt))]
        m = (N.cluster == -1) & (hav(N.lat, N.lon, N.lat[i], N.lon[i]) <= CLUSTER_M)
        N.loc[m, "cluster"] = c; c += 1
    H = N.groupby("cluster").agg(lat=("lat", "median"), lon=("lon", "median"), nights=("date", "size"),
                                 first=("date", "min"), last=("date", "max")).reset_index()
    H["home"] = H.nights >= 2
    return N, H


def stays(p, homes):
    d = load(p.sensor)
    d = d[(d.date >= p.start) & (d.date <= p.end + pd.Timedelta(days=1))].reset_index(drop=True)
    if not len(d) or not len(homes): return pd.DataFrame(), d
    near = np.zeros(len(d), bool)
    for h in homes.itertuples():
        near |= d.fix.values & (hav(d.lat.fillna(0).values, d.lon.fillna(0).values, h.lat, h.lon) <= RADIUS)
    state = np.where(~d.fix.values, "N", np.where(near, "H", "A"))
    # A runs shorter than A_RUN consecutive fixes are GPS jumps -> N
    fi = np.where(d.fix.values)[0]; s = state[fi]
    j = 0
    while j < len(fi):
        k = j
        while k < len(fi) and s[k] == s[j]: k += 1
        if s[j] == "A" and k - j < A_RUN: state[fi[j:k]] = "N"
        j = k
    d["state"] = state
    u = d.minute_unix.values; out = []; cur = None
    data_start = np.r_[True, np.diff(u) > GAP_MAX]
    for i in range(len(d)):
        if cur is not None and i > 0 and u[i] - u[i - 1] > GAP_MAX:
            cur.update(end=u[i - 1] + 60, end_censored=True); out.append(cur); cur = None
        if state[i] == "H" and cur is None:
            cur = dict(start=u[i], start_censored=bool(data_start[i]), end=np.nan, end_censored=False)
        elif state[i] == "A" and cur is not None:
            cur.update(end=u[i], end_censored=False); out.append(cur); cur = None
    if cur is not None:
        cur.update(end=u[-1] + 60, end_censored=True); out.append(cur)
    S = pd.DataFrame(out)
    if len(S):
        S["dur"] = S.end - S.start
        S["long"] = (S.dur >= MIN_STAY) | (S.end_censored & (S.dur >= CENS_MIN))
        S.insert(0, "pid", p.pid); S.insert(1, "sensor", p.sensor)
    return S, d


def episodes(p, S, d):
    """Away periods between long stays, split at data gaps > GAP_MAX (the export holds isolated study days).
    Shift episode = a block with >= 1 sustained-movement minute. home_return only when the block ends at the
    start of the next long stay (no gap) and that stay is not itself left-censored."""
    L = S[S.long].sort_values("start").reset_index(drop=True) if len(S) else S
    u = d.minute_unix.values
    cut = np.r_[0, np.where(np.diff(u) > GAP_MAX)[0] + 1, len(u)]
    starts = u[cut[:-1]]; ends = u[cut[1:] - 1] + 60          # data blocks [start, end)
    bounds = [(-np.inf, None)] + [(r.end, r) for r in L.itertuples()]
    out = []
    for k, (a0, prev) in enumerate(bounds):
        nxt = L.iloc[k] if k < len(L) else None
        a1 = nxt.start if nxt is not None else np.inf
        for b0, b1 in zip(starts, ends):
            lo, hi = max(a0, b0), min(a1, b1)
            if lo >= hi: continue
            seg = d[(d.minute_unix >= lo) & (d.minute_unix < hi)]
            sus = seg[seg.sustained]
            if not len(sus): continue
            ends_at_stay = nxt is not None and hi == a1 and not nxt.start_censored
            ret = a1 if ends_at_stay else np.nan
            out.append(dict(pid=p.pid, sensor=p.sensor, away_start=lo, start_censored=bool(lo == b0),
                            shift_start=sus.minute_unix.iloc[0], last_move=sus.minute_unix.iloc[-1],
                            home_return=ret, return_censored_short=bool(nxt.end_censored and nxt.dur < MIN_STAY) if ends_at_stay else np.nan,
                            return_stay_dur_min=(nxt.dur / 60) if ends_at_stay else np.nan,
                            block_end=b1, ends_at_data_gap=bool(hi == b1 and not ends_at_stay)))
    return pd.DataFrame(out)


def build_all(verbose=True):
    P = periods(); NL, HL, SL, EL = [], [], [], []
    for p in P.itertuples():
        N = night_locations(p)
        if len(N):
            N, H = cluster(N); H.insert(0, "pid", p.pid); H.insert(1, "sensor", p.sensor)
            NL.append(N); HL.append(H); homes = H[H.home]
        else:
            homes = pd.DataFrame()
        S, d = stays(p, homes)
        if len(S):
            SL.append(S); E = episodes(p, S, d)
            if len(E): EL.append(E)
    N = pd.concat(NL, ignore_index=True); H = pd.concat(HL, ignore_index=True)
    S = pd.concat(SL, ignore_index=True); E = pd.concat(EL, ignore_index=True)
    N.to_csv(os.path.join(W, "home_nights.csv"), index=False)        # internal: coordinates
    H.to_csv(os.path.join(W, "home_locations.csv"), index=False)     # internal: coordinates
    S.to_csv(os.path.join(W, "home_stays.csv"), index=False)
    E.to_csv(os.path.join(W, "shift_episodes.csv"), index=False)
    return P, N, H, S, E


def explore():
    """Outcome- and PM-blind diagnostics used to choose RADIUS and MIN_STAY."""
    P = periods(); jit = []; NL = []
    for p in P.itertuples():
        N = night_locations(p); d = load(p.sensor)
        for r in N[N.source == "night_fix"].itertuples() if len(N) else []:
            g = d[(d.date == r.date) & (d.hour < 4) & d.fix & ~d.moving]
            jit.append(hav(g.lat.values, g.lon.values, r.lat, r.lon))
        NL.append(N)
    jit = np.concatenate(jit); jit = jit[jit < 1000]   # nights split between places excluded
    print("overnight stationary fixes: distance from the night location (m), n =", len(jit))
    print({q: round(float(np.percentile(jit, q)), 1) for q in (50, 90, 95, 99, 99.5, 99.9)})
    N = pd.concat(NL, ignore_index=True)
    print("night locations:", len(N), N.source.value_counts().to_dict())
    dd = []
    for (pid, s), g in N.groupby(["pid", "sensor"]):
        g2, H = cluster(g)
        for c in g2.cluster.unique():
            if c == 0: continue
        big = H.sort_values("nights").iloc[-1]
        dd += list(hav(g.lat.values, g.lon.values, big.lat, big.lon))
    dd = np.array(dd)
    print("night location to main home (m): share <50, <100, <150, <200, <500:",
          [round(float((dd < x).mean()), 3) for x in (50, 100, 150, 200, 500)])
    print("between 150 and 2000 m:", int(((dd >= 150) & (dd < 2000)).sum()), "values", np.sort(dd[(dd >= 100) & (dd < 2000)]).round(0))


# ---------------------------------------------------------------- workday level and row labels
def crosses(node, ep, wd1):
    """Shift 'on' at 04:00: sustained movement in [04:00 - 2 h, 04:00) and sustained movement or the home return
    in [04:00, 04:00 + 2 h], all inside the same episode."""
    if ep.last_move < wd1 - 2 * 3600 or ep.shift_start >= wd1: return False
    d = load(int(node)); m = d.minute_unix.values; su = d.sustained.values
    inside = (m >= ep.shift_start) & (m <= ep.last_move)
    before = (su & inside & (m >= wd1 - 2 * 3600) & (m < wd1)).any()
    after = (su & inside & (m >= wd1) & (m <= wd1 + 2 * 3600)).any() or \
            (ep.home_return == ep.home_return and wd1 <= ep.home_return <= wd1 + 2 * 3600)
    return bool(before and after)


def workday_table(E=None):
    """Per (pid, sensor, workday 04:00-04:00): first shift start, day's home return (return of the last shift
    episode that starts in the workday), last sustained movement, 04:00 crossing, midday long home stays."""
    if E is None: E = pd.read_csv(os.path.join(W, "shift_episodes.csv"))
    E = E.copy()
    E["workday"] = (pd.Series(unix_to_local(E.shift_start), index=E.index) - pd.Timedelta(hours=4)).dt.normalize()
    out = []
    for (pid, s, wd), g in E.sort_values("shift_start").groupby(["pid", "sensor", "workday"]):
        wd1 = local_to_unix(wd) + 4 * 3600 + 86400; last = g.iloc[-1]
        hr, lm = last.home_return, last.last_move
        out.append(dict(pid=pid, sensor=s, workday=wd, ep_shift_start=g.shift_start.iloc[0], home_return=hr,
                        last_move=lm, return_censored_short=last.return_censored_short, n_episodes=len(g),
                        crosses_0400=crosses(s, last, wd1), wd_end=wd1))
    return pd.DataFrame(out)


def label(T, pid, node, wd, t):
    """protocol_slot, reason, home_return (unix or nan), crosses (True/False/nan) for one entry at time t."""
    if node != node: return "unclear", "no_sensor", np.nan, np.nan
    wd = pd.Timestamp(wd); key = (pid, int(node))
    cur = T.get(key + (wd,)); prev = T.get(key + (wd - pd.Timedelta(days=1),))
    prev_on = prev is not None and t >= prev["ep_shift_start"] and (
        (prev["home_return"] == prev["home_return"] and t < prev["home_return"]) or
        (prev["home_return"] != prev["home_return"] and t <= prev["last_move"]))
    if cur is None:
        if prev_on: return "during_shift", "previous_workday_shift", np.nan, np.nan
        return "unclear", "no_shift_episode", np.nan, np.nan
    hr, cr = cur["home_return"], cur["crosses_0400"]
    if t < cur["ep_shift_start"]:
        if prev_on: return "during_shift", "previous_workday_shift", hr, cr
        return "before_shift", "", hr, cr
    if hr == hr:
        return ("during_shift", "", hr, cr) if t < hr else ("after_home", "", hr, cr)
    if t <= cur["last_move"]: return "during_shift", "", hr, cr
    return "unclear", "return_not_observed", hr, cr


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--explore", action="store_true"); a = ap.parse_args()
    if a.explore:
        explore()
    else:
        P, N, H, S, E = build_all()
        print("person-periods", len(P), "| night locations", len(N), "| homes", int(H.home.sum()),
              "| periods with >1 home", int((H[H.home].groupby(["pid", "sensor"]).size() > 1).sum()),
              "| stays", len(S), "long", int(S.long.sum()), "| shift episodes", len(E), "with return", int(E.home_return.notna().sum()))
