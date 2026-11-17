#!/usr/bin/env python3
"""Refuge addendum A1 (refuge/SPEC.md §A1, 2026-11-12): temperature-drop events.

Stationary on-shift episodes (>= 5 min, > 300 m from home) where the device temperature
falls vs the 10 min before. A ladder of thresholds; PM during the event vs (a) own
10 min before and (b) the concurrent same-city moving-rider reference, adjusted for the
unit's own baseline ratio to that reference. Ratios and % only (raw PM).

  python3 -I scripts/refuge_drop_events.py --blind      # temperature only, no PM read
  python3 -I scripts/refuge_drop_events.py [--pm-col pm]
"""
import argparse, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import refuge_analysis as ra  # noqa: E402

ROOT = ra.ROOT
SEED = 20261112
NBOOT = 4000
LADDER = [2.0, 3.0, 4.0, 5.0]   # fixed in SPEC §A1 after --blind
PRIMARY = 3.0
SKIP = 2                        # minutes skipped at the start of an event


def episodes(D):
    rows, ons = [], []
    for n, d in D.items():
        for pid, dp in d.groupby("person"):
            night = dp[(dp.hour < 4) & dp.lat.notna()]
            hk = (night.lat.round(3).astype(str) + "," + night.lon.round(3).astype(str)).mode()
            hlat, hlon = map(float, hk.iloc[0].split(","))
            for wd, g in dp.groupby("wd"):
                mvi = g.index[g.moving.values]
                if len(mvi) < 10:
                    continue
                s = d.loc[mvi[0]:mvi[-1]]
                ons.append(s.index.values)
                lat, lon, um = s.lat.values, s.lon.values, s.minute_unix.values
                i, L = 0, len(s)
                while i < L:
                    if np.isnan(lat[i]):
                        i += 1; continue
                    alat, alon, j, last = lat[i], lon[i], i, i
                    while j + 1 < L and um[j + 1] - um[j] == 60:
                        if np.isnan(lat[j + 1]):
                            if j + 1 - last > 15: break
                            j += 1; continue
                        if ra.hav(alat, alon, lat[j + 1], lon[j + 1]) <= 50:
                            j += 1; last = j
                        else:
                            break
                    j = last
                    if j - i + 1 >= 5 and ra.hav(alat, alon, hlat, hlon) > 300:
                        rows.append(dict(node=n, person=pid, wd=wd, city=g.city.iloc[0],
                                         r0=s.index[i], r1=s.index[j], start=um[i], L=j - i + 1))
                    i = j + 1
    return pd.DataFrame(rows)


def temp_features(ev, D):
    out = []
    for r in ev.itertuples():
        d = D[r.node]
        core = d.loc[r.r0 + SKIP:r.r1]
        pre = d.loc[max(r.r0 - 10, 0):r.r0 - 1]
        pre = pre[(pre.minute_unix >= r.start - 600) & (pre.person == r.person)]
        ok = pre.temp.notna().sum() >= 5 and core.temp.notna().sum() >= 3
        own = pre.temp.median() - core.temp.median() if ok else np.nan
        refd = (pre.ref_temp.median() - core.ref_temp.median()
                if pre.ref_temp.notna().sum() >= 3 and core.ref_temp.notna().sum() >= 2 else np.nan)
        out.append(dict(drop=own, ref_drop=refd,
                        hour=pd.Timestamp(r.start + 7 * 3600, unit="s").hour,
                        nogps=int(d.loc[r.r0:r.r1].lat.isna().sum())))
    return pd.concat([ev.reset_index(drop=True), pd.DataFrame(out)], axis=1)


def pm_features(ev, D, k_unit, shift=0.0):
    out = []
    for r in ev.itertuples():
        d = D[r.node]
        core = d.loc[r.r0 + SKIP:r.r1]
        pre = d.loc[max(r.r0 - 10, 0):r.r0 - 1]
        pre = pre[(pre.minute_unix >= r.start - 600) & (pre.person == r.person)]
        pm, w = core.pm.values + shift, core.n_pm.fillna(0).values
        rp = core.ref_pm.values + shift
        cov = np.isfinite(core.pm.values).mean()
        both = np.isfinite(pm) & np.isfinite(rp)
        pre_pm = ra.wmean(pre.pm.values + shift, pre.n_pm.fillna(0).values) if pre.pm.notna().sum() >= 5 else np.nan
        ev_pm = ra.wmean(pm, w) if cov >= 0.5 else np.nan
        ra_ = ev_pm / pre_pm if (np.isfinite(ev_pm) and pre_pm and pre_pm > 0) else np.nan
        k = k_unit[(r.node, r.person)][shift]
        rb = (ra.wmean(pm[both], w[both]) / rp[both].mean() / k) if (cov >= 0.5 and both.mean() >= 0.5) else np.nan
        out.append(dict(ratio_a=ra_, ratio_b=rb))
    return pd.concat([ev.reset_index(drop=True)[[c for c in ev.columns if c not in ("ratio_a", "ratio_b")]],
                      pd.DataFrame(out)], axis=1)


def unit_k(D, ons_rows, shifts=(0.0, -5.0, 5.0)):
    """Unit's baseline own/ref ratio: ratio of means over its on-shift MOVING minutes with both values."""
    k = {}
    for n, d in D.items():
        sub = d.loc[np.intersect1d(ons_rows[n], d.index)]
        sub = sub[sub.moving]
        for pid, g in sub.groupby("person"):
            m = g.pm.notna() & g.ref_pm.notna()
            k[(n, pid)] = {s: (g.pm[m] + s).mean() / (g.ref_pm[m] + s).mean() for s in shifts}
    return k


def pool_city(ev, col, boot=True):
    """Among riders with >= 1 event: equal-weight mean of per-rider mean log ratio."""
    e = ev[ev[col] > 0]
    pr = e.groupby("person")[col].apply(lambda x: np.log(x).mean())
    if len(pr) == 0:
        return dict(k=0, est=np.nan, lo=np.nan, hi=np.nan)
    x = pr.values
    res = dict(k=len(x), est=np.exp(x.mean()), lo=np.nan, hi=np.nan)
    if boot and len(x) >= 3:
        rng = np.random.default_rng(SEED)
        bs = np.exp([x[rng.integers(0, len(x), len(x))].mean() for _ in range(NBOOT)])
        res["lo"], res["hi"] = np.percentile(bs, 2.5), np.percentile(bs, 97.5)
    return res


def traj(ev, D, var):
    """Mean trajectory -15..+30 min relative to event start: temp difference vs pre-10 median (°C),
    or log PM ratio vs pre-10 mean."""
    M = []
    for r in ev.itertuples():
        d = D[r.node].set_index("minute_unix")
        m = r.start + 60 * np.arange(-15, 31)
        s = d.reindex(m)
        pre = d.reindex(r.start + 60 * np.arange(-10, 0))
        if var == "temp":
            M.append(s.temp.values - pre.temp.median())
        else:
            b = pre.pm.mean()
            M.append(np.log(s.pm.values / b) if b and b > 0 else np.full(len(m), np.nan))
    M = np.array(M, dtype=float)
    M[~np.isfinite(M)] = np.nan
    return np.nanmean(M, axis=0) if len(M) else np.full(46, np.nan)


def svg_traj(curves, path):
    Wd, H = 620, 330
    x0, x1 = 60, 290
    X = lambda t, off: off + x0 + (t + 15) / 45 * (x1 - x0)
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{Wd}" height="{H}" font-family="sans-serif" font-size="11">',
         '<rect width="100%" height="100%" fill="white"/>',
         f'<text x="10" y="16" font-size="12" font-weight="bold">Temperature-drop events (drop ≥ {PRIMARY:.0f} °C): mean trajectory around event start</text>']
    cols = {"BKK": "#3b6ea5", "CMI": "#c0392b"}
    panels = [("temp", "Device temp − pre-event median (°C)", -6, 2, 0), ("pm", "PM ÷ pre-event mean (log scale)", np.log(0.5), np.log(1.5), 300)]
    t = np.arange(-15, 31)
    for var, lab, lo, hi, off in panels:
        Y = lambda v: 290 - (v - lo) / (hi - lo) * 250
        s.append(f'<text x="{off+x0}" y="34">{lab}</text>')
        s.append(f'<line x1="{X(0,off):.1f}" y1="40" x2="{X(0,off):.1f}" y2="290" stroke="#888" stroke-dasharray="3,3"/>')
        zero = 0 if var == "temp" else 0.0
        s.append(f'<line x1="{X(-15,off):.1f}" y1="{Y(zero):.1f}" x2="{X(30,off):.1f}" y2="{Y(zero):.1f}" stroke="#aaa"/>')
        ticks = [-6, -4, -2, 0, 2] if var == "temp" else [np.log(v) for v in (0.5, 0.75, 1.0, 1.25, 1.5)]
        for v in ticks:
            labv = f"{v:.0f}" if var == "temp" else f"{np.exp(v):.2f}"
            s.append(f'<text x="{off+x0-5}" y="{Y(v)+4:.1f}" text-anchor="end">{labv}</text>')
        for tt in (-15, 0, 15, 30):
            s.append(f'<text x="{X(tt,off):.1f}" y="306" text-anchor="middle">{tt}</text>')
        for c, cur in curves.items():
            v = np.clip(cur[var], lo, hi)
            pts = " ".join(f"{X(a,off):.1f},{Y(b):.1f}" for a, b in zip(t, v) if np.isfinite(b))
            s.append(f'<polyline points="{pts}" fill="none" stroke="{cols[c]}" stroke-width="2"/>')
    s.append(f'<text x="{x0}" y="324">minutes from event start (events last ≥ 5 min; later minutes include riders who left)</text>')
    for i, (c, cur) in enumerate(curves.items()):
        s.append(f'<text x="{480}" y="{50+14*i}" fill="{cols[c]}">{c} (n={cur["n"]})</text>')
    s.append("</svg>")
    open(path, "w").write("\n".join(s))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pm-col", default="pm")
    ap.add_argument("--blind", action="store_true")
    ap.add_argument("--out", default=os.path.join(ROOT, "refuge"))
    A = ap.parse_args()
    T, F = os.path.join(A.out, "tables"), os.path.join(A.out, "figures")
    coh = pd.read_csv(os.path.join(ROOT, "foundation", "cohort.csv"), dtype={"sensor": str})
    coh = coh[coh.role == "rider"].set_index("pid")
    coh["enrol_start"] = pd.to_datetime(coh.enrol_start); coh["enrol_end"] = pd.to_datetime(coh.enrol_end)
    ra.COH = coh
    D = ra.load(A.pm_col)
    if A.blind:
        for d in D.values():
            d["pm"] = np.nan  # PM never used in blind mode
    D = ra.references(D)
    ev = temp_features(episodes(D), D)
    ev = ev[ev["drop"].notna()].reset_index(drop=True)
    L = []
    p = lambda *x: L.append(" ".join(str(y) for y in x))
    p(f"episodes >=5 min, >300 m from home, with temp: {len(ev)}; riders {ev.person.nunique()}")
    q = ev["drop"].quantile([.5, .75, .9, .95, .99])
    p("drop quantiles:", q.round(2).to_dict())
    for th in LADDER:
        e = ev[ev["drop"] >= th]
        p(f"drop>={th}: events {len(e)} ({100*len(e)/len(ev):.1f}%), riders {e.person.nunique()}, "
          f"by city {e.city.value_counts().to_dict()}, riders by city {e.groupby('city').person.nunique().to_dict()}")
    if A.blind:
        h, edges = np.histogram(ev["drop"].clip(-8, 10), np.arange(-8, 10.5, 1))
        for a, c in zip(edges[:-1], h):
            p(f"  {a:5.1f} {c:5d}")
        e = ev[ev["drop"] >= PRIMARY]
        p(f"primary events: median length {e.L.median()} min; ref temp drop median {e.ref_drop.median():.2f} "
          f"(all episodes {ev.ref_drop.median():.2f}); share ref_drop >= own/2: {(e.ref_drop >= e['drop']/2).mean():.3f}")
        p("hour (primary):", e.hour.value_counts().sort_index().to_dict())
        print("\n".join(L)); return

    ons_rows = {}
    # on-shift rows per node (same rule as episodes)
    for n, d in D.items():
        rows = []
        for (pid, wd), g in d.groupby(["person", "wd"]):
            mvi = g.index[g.moving.values]
            if len(mvi) >= 10:
                rows.append(np.arange(mvi[0], mvi[-1] + 1))
        ons_rows[n] = np.concatenate(rows) if rows else np.array([], int)
    k = unit_k(D, ons_rows)
    ev = pm_features(ev, D, k)
    ev["excess_drop"] = ev["drop"] - ev.ref_drop
    ev.drop(columns=["r0", "r1"]).to_csv(os.path.join(T, "drop_events.csv"), index=False)

    p("\n## Ladder (point estimates among riders with >= 1 event; per-event median [IQR] of % change)")
    lad = []
    for th in [None, 0.5] + LADDER:
        e = ev[(ev["drop"] < 0.5)] if th is None else ev[ev["drop"] >= th] if th != 0.5 else ev[ev["drop"].between(0.5, 2, inclusive="left")]
        lab = "non-cooling (<0.5)" if th is None else ("0.5-2" if th == 0.5 else f">={th}")
        for c in ["BKK", "CMI", "all"]:
            ec = e if c == "all" else e[e.city == c]
            for col in ["ratio_a", "ratio_b"]:
                v = ec[col].dropna()
                boot = (th == PRIMARY and c != "all")
                r = pool_city(ec, col, boot=boot)
                lad.append(dict(stratum=lab, city=c, ratio=col, events=len(v), riders=r["k"],
                                rider_pooled=r["est"], lo=r["lo"], hi=r["hi"],
                                ev_median=v.median(), ev_q25=v.quantile(.25), ev_q75=v.quantile(.75)))
    lad = pd.DataFrame(lad); lad.to_csv(os.path.join(T, "drop_ladder.csv"), index=False)
    p(lad.round(3).to_string(index=False))

    e3 = ev[ev["drop"] >= PRIMARY]
    p("\n## Alternative explanations (primary events)")
    p(f"city reference also cooled >= half the own drop: {(e3.ref_drop >= e3['drop']/2).mean():.3f} (n with ref {e3.ref_drop.notna().sum()})")
    ex = e3[e3.excess_drop >= PRIMARY]
    for c in ["BKK", "CMI"]:
        for col in ["ratio_a", "ratio_b"]:
            r = pool_city(ex[ex.city == c], col, boot=False)
            p(f"  excess drop (own - city) >= {PRIMARY}: {c} {col} rider-pooled {r['est']:.3f} (k {r['k']}, events {ex[(ex.city==c)][col].notna().sum()})")
    e3h = e3.assign(eve=e3.hour >= 17)
    for c in ["BKK", "CMI"]:
        for eve in [False, True]:
            r = pool_city(e3h[(e3h.city == c) & (e3h.eve == eve)], "ratio_a", boot=False)
            p(f"  {c} {'17:00+' if eve else 'before 17:00'} ratio_a rider-pooled {r['est']:.3f} (k {r['k']})")
    p(f"no-GPS minutes share: primary events {e3.nogps.sum()/e3.L.sum():.3f}; non-cooling {ev[ev['drop']<0.5].nogps.sum()/ev[ev['drop']<0.5].L.sum():.3f}")

    p("\n## Intercept sensitivity (primary threshold, rider-pooled)")
    for sh in (-5.0, 5.0):
        e2 = pm_features(e3, D, k, shift=sh)
        for c in ["BKK", "CMI"]:
            ra_ = pool_city(e2[e2.city == c], "ratio_a", boot=False)["est"]
            rb_ = pool_city(e2[e2.city == c], "ratio_b", boot=False)["est"]
            p(f"  shift {sh:+}: {c} ratio_a {ra_:.3f}, ratio_b {rb_:.3f}")

    curves = {}
    for c in ["BKK", "CMI"]:
        ec = e3[e3.city == c]
        curves[c] = dict(temp=traj(ec, D, "temp"), pm=traj(ec, D, "pm"), n=len(ec))
        p(f"trajectory {c}: temp at +5/+10/+20 = {curves[c]['temp'][[20,25,35]].round(2)}; PM ratio = {np.exp(curves[c]['pm'][[20,25,35]]).round(3)}")
    svg_traj(curves, os.path.join(F, "fig3_drop_event_trajectory.svg"))
    open(os.path.join(T, "drop_events_log.txt"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
