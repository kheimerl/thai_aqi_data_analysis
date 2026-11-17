#!/usr/bin/env python3
"""Refuge addendum P1 (refuge/SPEC.md §P1, 2026-11-17): review comment 1.

Where was the helmet sensor during long stops, and how far can that move the
rider/office ratio (E1, audited 1.72 BKK / 1.63 CMI) and each rider's on-shift dose?
Scenarios for long-stop minutes: (a) measured, (b) removed, (c) plausible indoor value
(lower of own home-night level and same-city office level that hour), (d) same-city
on-road reference that hour. Ratios only (raw PM).

  python3 -I scripts/refuge_waiting_bounds.py [--pm-col pm] [--out refuge]
"""
import argparse, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import refuge_analysis as ra  # noqa: E402

ROOT = ra.ROOT
OFFICE_NODES = [6009, 6010, 6033, 6007]   # 6006 dropped (S2 §7); 6008 not office
PM_DROP_INDOOR = 0.80                     # SPEC §P1.2


def hourly_valid():
    """S2 rules: >= 10 PM minutes, pm_mean != 0, person assigned; city by GPS (lat > 17 -> CMI)."""
    h = pd.read_csv(os.path.join(ROOT, "foundation", "pm_hourly_by_node.csv"),
                    usecols=["node", "person", "hour_unix", "pm_mean", "minutes_pm", "lat_median"])
    nodes = pd.read_csv(os.path.join(ROOT, "foundation", "nodes.csv"))
    home_city = dict(zip(nodes.node, nodes.city))
    h["city"] = np.where(h.lat_median > 17, "CMI", np.where(h.lat_median.notna(), "BKK", h.node.map(home_city)))
    v = h[(h.minutes_pm >= 10) & (h.pm_mean != 0) & h.pm_mean.notna()]
    return v, home_city


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pm-col", default="pm")
    ap.add_argument("--out", default=os.path.join(ROOT, "refuge"))
    A = ap.parse_args()
    T = os.path.join(A.out, "tables"); os.makedirs(T, exist_ok=True)
    coh = pd.read_csv(os.path.join(ROOT, "foundation", "cohort.csv"), dtype={"sensor": str})
    coh = coh[coh.role == "rider"].set_index("pid")
    coh["enrol_start"] = pd.to_datetime(coh.enrol_start); coh["enrol_end"] = pd.to_datetime(coh.enrol_end)
    ra.COH = coh

    D = ra.references(ra.load(A.pm_col))
    st0, ons = ra.find_stops(D)
    st = ra.classify(ra.stop_metrics(st0, D))
    L = []
    p = lambda *x: L.append(" ".join(str(y) for y in x))

    # ---------- helper values per minute ----------
    hv, home_city = hourly_valid()
    off = hv[hv.node.isin(OFFICE_NODES)].groupby(["city", "hour_unix"]).pm_mean.mean()
    for n, d in D.items():
        d["hour_unix"] = d.minute_unix // 3600 * 3600
        d["date"] = pd.to_datetime(d.minute_unix + 7 * 3600, unit="s").dt.normalize()
        # home-night level per person-date (00-04 local, within 300 m of home)
        d["home_val"] = np.nan
        for pid, dp in d.groupby("person"):
            night = dp[(dp.hour < 4) & dp.lat.notna()]
            hk = (night.lat.round(3).astype(str) + "," + night.lon.round(3).astype(str)).mode()
            hlat, hlon = map(float, hk.iloc[0].split(","))
            nn = dp[(dp.hour < 4) & dp.pm.notna()]
            # minutes without GPS at night count as home (sensor indoors at home)
            near = nn.lat.isna() | (ra.hav(nn.lat.fillna(hlat), nn.lon.fillna(hlon), hlat, hlon) <= 300)
            nn = nn[near]
            g = nn.groupby("date").apply(lambda x: pd.Series(dict(v=ra.wmean(x.pm.values, x.n_pm.fillna(0).values), m=len(x))),
                                         include_groups=False)
            fallback = g.v.median() if len(g) else np.nan
            good = g[g.m >= 30].v
            d.loc[dp.index, "home_val"] = dp.date.map(good).fillna(fallback).values
        d["off_val"] = [off.get((c, hu), np.nan) for c, hu in zip(d.city, d.hour_unix)]
        d["indoor_val"] = np.fmin(d.home_val, d.off_val)   # fmin ignores NaN -> home alone if no office
        d["road_val"] = d.hour_unix.map(d.groupby("hour_unix").ref_pm.mean())

    # long-stop masks
    masks = {}
    for thr in (10, 30):
        for n, d in D.items():
            m = np.zeros(len(d), bool)
            for r in st[(st.node == n) & (st.L >= thr)].itertuples():
                m[r.r0:r.r1 + 1] = True   # D[n] has a RangeIndex, so rows == positions
            masks[(thr, n)] = m

    def scen_pm(d, m, s):
        pm = d.pm.values.copy(); w = d.n_pm.fillna(0).values.astype(float).copy()
        if s == "b":
            pm[m] = np.nan; w[m] = 0
        elif s == "c":
            v = d.indoor_val.values; sel = m & np.isfinite(v) & np.isfinite(pm); pm[sel] = v[sel]
        elif s == "d":
            v = d.road_val.values; sel = m & np.isfinite(v) & np.isfinite(pm); pm[sel] = v[sel]
        return pm, w

    # ---------- 1. how much it matters ----------
    p("## 1. Share of on-shift time and dose in long stops (>300 m from home)")
    rows = []
    for (n, pid), g in ons.groupby(["node", "person"]):
        d = D[n].loc[g.row.values]
        pos = g.row.values
        w = d.n_pm.fillna(0).values; pmw = np.nan_to_num(d.pm.values) * w
        r = dict(node=n, person=pid, city=d.city.mode().iloc[0], on_min=len(d), dose=pmw.sum())
        for thr in (10, 30):
            m = masks[(thr, n)][pos]
            r[f"min_ge{thr}"] = m.sum(); r[f"dose_ge{thr}"] = pmw[m].sum()
        rows.append(r)
    sh = pd.DataFrame(rows).groupby(["person", "city"], as_index=False).sum(numeric_only=True)
    for thr in (10, 30):
        sh[f"time_share_ge{thr}"] = sh[f"min_ge{thr}"] / sh.on_min
        sh[f"dose_share_ge{thr}"] = sh[f"dose_ge{thr}"] / sh.dose
    out1 = sh[["person", "city", "on_min"] + [c for c in sh.columns if "share" in c]]
    out1.to_csv(os.path.join(T, "p1_waiting_share_per_rider.csv"), index=False)
    for c, g in list(sh.groupby("city")) + [("all", sh)]:
        p(f"  {c}: riders {len(g)}; time >=10 median {100*g.time_share_ge10.median():.1f}% (range {100*g.time_share_ge10.min():.1f}-{100*g.time_share_ge10.max():.1f}), "
          f"pooled {100*g.min_ge10.sum()/g.on_min.sum():.1f}%; dose >=10 median {100*g.dose_share_ge10.median():.1f}%, pooled {100*g.dose_ge10.sum()/g.dose.sum():.1f}% | "
          f">=30: time median {100*g.time_share_ge30.median():.1f}%, dose median {100*g.dose_share_ge30.median():.1f}%")

    # ---------- 2. carried in? ----------
    p("\n## 2. Probably-indoor long stops (temp-marked AND PM ratio_a <= 0.80)")
    tempmark = (st.type == "refuge") | (st.drop_pre >= 3)
    classifiable = st.dT.notna() | st.drop_pre.notna()
    st["indoor"] = np.where(~classifiable, np.nan, (tempmark & (st.ratio_a <= PM_DROP_INDOOR)).astype(float))
    ic = []
    for thr in (10, 30):
        s = st[(st.L >= thr) & st.indoor.notna()]
        g = s.groupby(["person", "city"]).agg(stops=("indoor", "size"), indoor=("indoor", "sum")).reset_index()
        g["share"] = g.indoor / g.stops; g["min_L"] = thr
        ic.append(g)
        p(f"  >= {thr} min: stops {len(s)}, probably indoor {int(s.indoor.sum())} ({100*s.indoor.mean():.1f}%); "
          f"per-rider share median {100*g.share.median():.1f}%, max {100*g.share.max():.1f}%; "
          f"riders with >= 10% indoor: {sorted(g[g.share >= 0.10].person.tolist())}")
        for c, gc in s.groupby("city"):
            p(f"     {c}: {100*gc.indoor.mean():.1f}% of {len(gc)} stops")
    ic = pd.concat(ic); ic.to_csv(os.path.join(T, "p1_indoor_share_per_rider.csv"), index=False)

    # ---------- 3. bounds ----------
    p("\n## 3. Bounds")
    e1 = pd.read_csv(os.path.join(ROOT, "calibration", "tables", "e1_matched_hours.csv"))
    e1 = e1[e1.office != "O05"].copy()
    e1["hour_unix"] = ((pd.to_datetime(e1.hour_local) - pd.Timedelta(hours=7)) - pd.Timestamp("1970-01-01")) // pd.Timedelta(seconds=1)
    rv = hv[hv.node.isin(ra.NODES)][["node", "hour_unix", "city", "pm_mean"]]
    # node-hours covered by this analysis' enrolled minutes; other valid S2 hours keep S2's value
    # (they lie outside cohort enrolment, e.g. R07-R10 Jan 9-12, so they contain no detected stop)
    cov = pd.concat([pd.DataFrame({"node": n, "hour_unix": d.hour_unix.unique()}) for n, d in D.items()])
    cov["in_D"] = True
    rv = rv.merge(cov, on=["node", "hour_unix"], how="left")
    rv["in_D"] = rv.in_D.fillna(False).astype(bool)
    res, dose_rows = [], []
    base_dose = {}
    for thr in (10, 30):
        for s in "abcd":
            hm = []
            for n, d in D.items():
                pm, w = scen_pm(d, masks[(thr, n)], s)
                ok = np.isfinite(pm) & (w > 0)
                t = pd.DataFrame({"hour_unix": d.hour_unix.values[ok], "sw": pm[ok] * w[ok], "w": w[ok]})
                agg = t.groupby("hour_unix").agg(sw=("sw", "sum"), w=("w", "sum"), mins=("w", "size"))
                agg["node"] = n; hm.append(agg.reset_index())
                # on-shift dose per person
                for pid, g in ons[ons.node == n].groupby("person"):
                    pos = g.row.values
                    val = ra.wmean(pm[pos], w[pos])
                    key = (thr, pid, n)
                    if s == "a": base_dose[key] = (val, d.city.loc[pos].mode().iloc[0])
                    dose_rows.append(dict(min_L=thr, scenario=s, person=pid, node=n, val=val))
            hm = pd.concat(hm)
            hm = hm[(hm.mins >= 10) & (hm.sw > 0)]
            hm["pm_s"] = hm.sw / hm.w
            r = rv.merge(hm[["node", "hour_unix", "pm_s"]], on=["node", "hour_unix"], how="left")
            r["pm_s"] = np.where(r.in_D, r.pm_s, r.pm_mean)
            r = r[r.pm_s.notna()]
            comp = r.groupby(["city", "hour_unix"]).pm_s.agg(["mean", "size"]).reset_index()
            m = e1.merge(comp, on=["city", "hour_unix"], how="left")
            m = m[m["size"] >= 2]
            for c, g in m.groupby("city"):
                res.append(dict(min_L=thr, scenario=s, city=c, hours=len(g),
                                ratio=g["mean"].sum() / g.office_raw.sum(),
                                pct_rider_higher=(g["mean"] > g.office_raw).mean(),
                                repro_maxdiff=(g["mean"] - g.rider_mean_raw).abs().max() if s == "a" else np.nan))
    res = pd.DataFrame(res); res.to_csv(os.path.join(T, "p1_bounds_e1.csv"), index=False)
    p(res.round(3).to_string(index=False))
    dr = pd.DataFrame(dose_rows)
    dr["base"] = [base_dose[(a, b, c)][0] for a, b, c in zip(dr.min_L, dr.person, dr.node)]
    dr["city"] = [base_dose[(a, b, c)][1] for a, b, c in zip(dr.min_L, dr.person, dr.node)]
    # person-level (R22 spans two nodes): weight by node? use mean of node ratios weighted equally
    dr["rel"] = dr.val / dr.base
    pr = dr.groupby(["min_L", "scenario", "person", "city"]).rel.mean().reset_index()
    pr.to_csv(os.path.join(T, "p1_bounds_dose_per_rider.csv"), index=False)
    p("\n  on-shift dose relative to measured (per rider):")
    p(pr.groupby(["min_L", "scenario", "city"]).rel.describe()[["count", "min", "50%", "max"]].round(3).to_string())

    open(os.path.join(T, "p1_log.txt"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
