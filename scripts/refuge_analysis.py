#!/usr/bin/env python3
"""Refuge analysis (refuge/SPEC.md, 2026-11-12). Exposure only; no PVT/PEF.

How much do refuge-type stops (device cooler than the concurrent road reference)
lower raw PM2.5 relative to outdoor stops, and what share of on-shift exposure do
they avert? Ratios and % only (raw PM, rule 7).

Run: python3 -I scripts/refuge_analysis.py [--pm-col pm] [--out refuge]
"""
import argparse, os, sys
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
W = os.path.join(ROOT, "foundation", "_work")
NODES = list(range(6011, 6031)) + [6032]
SEED = 20261112
NBOOT = 4000
TH_REF, TH_OUT = -3.0, -1.0
P2R = {"P1": "R04", "P2": "R21", "P3": "R14", "P4": "R16", "P5": "R12", "P6": "R05", "P7": "R02",
       "P8": "R20", "P9": "R17", "P10": "R18", "P11": "R22", "P12": "R01", "P13": "R06"}


def hav(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2
    return 2 * 6371000.0 * np.arcsin(np.sqrt(a))


def person_of(node, day):
    """day = workday (Timestamp, normalized). Returns person or None (outside enrolment)."""
    c = COH
    if node == 6021:
        pid = "R11" if day <= pd.Timestamp("2026-02-26") else "R12"
    elif node == 6022:
        pid = "R13" if day <= pd.Timestamp("2026-03-25") else "R14"
    elif node in (6029, 6032):
        pid = "R22"
    else:
        pid = c.index[c.sensor == str(node)][0]
    s, e = c.loc[pid, "enrol_start"], c.loc[pid, "enrol_end"]
    return pid if s <= day <= e else None


def load(pm_col):
    out = {}
    for n in NODES:
        d = pd.read_csv(os.path.join(W, f"minute_{n}.csv"),
                        usecols=["minute_unix", "n_pm", pm_col, "temp", "lat", "lon", "km_step"])
        d = d.rename(columns={pm_col: "pm"}).sort_values("minute_unix").reset_index(drop=True)
        d.loc[d.n_pm.fillna(0) <= 0, "pm"] = np.nan
        loc = pd.to_datetime(d.minute_unix + 7 * 3600, unit="s")
        d["hour"] = loc.dt.hour
        d["wd"] = (loc - pd.Timedelta(hours=4)).dt.normalize()
        d["moving"] = d.km_step.between(0.1, 2)
        pmap = {wd: person_of(n, wd) for wd in d.wd.unique()}
        d["person"] = d.wd.map(pmap)
        city = d.groupby("wd").lat.median().apply(lambda x: "CMI" if x > 15.5 else ("BKK" if x == x else None))
        d["city"] = d.wd.map(city)
        d["node"] = n
        out[n] = d[d.person.notna()].reset_index(drop=True)
    return out


def references(D):
    """Per node: per-minute median PM and temp of OTHER same-city moving rider nodes (>= 2)."""
    for var in ["pm", "temp"]:
        wide = {}
        for n, d in D.items():
            for c in ["BKK", "CMI"]:
                s = d.loc[d.moving & (d.city == c)].set_index("minute_unix")[var]
                wide[(c, n)] = s
        for c in ["BKK", "CMI"]:
            M = pd.concat({n: s for (cc, n), s in wide.items() if cc == c}, axis=1)
            for n, d in D.items():
                others = M.drop(columns=n, errors="ignore")
                ok = others.notna().sum(axis=1) >= 2
                ref = others.median(axis=1).where(ok)
                col = f"ref_{var}"
                v = d.minute_unix.map(ref)
                if col not in d:
                    d[col] = np.nan
                sel = d.city == c
                d.loc[sel, col] = v[sel]
    return D


def wmean(pm, w):
    m = np.isfinite(pm) & (w > 0)
    return (pm[m] * w[m]).sum() / w[m].sum() if m.any() else np.nan


def find_stops(D):
    rows, onshift = [], []
    for n, d in D.items():
        for pid, dp in d.groupby("person"):
            night = dp[(dp.hour < 4) & dp.lat.notna()]
            hk = (night.lat.round(3).astype(str) + "," + night.lon.round(3).astype(str)).mode()
            hlat, hlon = map(float, hk.iloc[0].split(","))
            mv = dp[dp.moving]
            off = (mv.temp - mv.ref_temp).median()
            for wd, g in dp.groupby("wd"):
                mvi = g.index[g.moving.values]
                if len(mvi) < 10:
                    continue
                s = d.loc[mvi[0]:mvi[-1]]
                onshift.append(pd.DataFrame({"node": n, "person": pid, "minute_unix": s.minute_unix.values,
                                             "row": s.index.values}))
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
                        if hav(alat, alon, lat[j + 1], lon[j + 1]) <= 50:
                            j += 1; last = j
                        else:
                            break
                    j = last
                    if j - i + 1 >= 10 and hav(alat, alon, hlat, hlon) > 300:
                        rows.append(dict(node=n, person=pid, wd=wd, city=g.city.iloc[0],
                                         r0=s.index[i], r1=s.index[j], start=um[i], L=j - i + 1,
                                         nogps=int(np.isnan(lat[i:j + 1]).sum()), off=off))
                    i = j + 1
    return pd.DataFrame(rows), pd.concat(onshift, ignore_index=True)


def stop_metrics(st, D, shift=0.0):
    out = []
    for r in st.itertuples():
        d = D[r.node]
        g = d.loc[r.r0:r.r1]
        core = g.iloc[3:]
        pre = d[(d.minute_unix >= r.start - 600) & (d.minute_unix < r.start) & (d.person == r.person)]
        pm = core.pm.values + shift
        rp = core.ref_pm.values + shift
        w = core.n_pm.fillna(0).values
        pm_cov = np.mean(np.isfinite(core.pm.values))
        both = np.isfinite(pm) & np.isfinite(rp)
        ref_cov_t = np.mean(np.isfinite(core.ref_temp.values))
        dT = np.nanmedian((core.temp - core.ref_temp).values) - r.off if ref_cov_t >= 0.5 else np.nan
        pre_pm = wmean(pre.pm.values + shift, pre.n_pm.fillna(0).values) if pre.pm.notna().sum() >= 5 else np.nan
        stop_pm = wmean(pm, w) if pm_cov >= 0.5 else np.nan
        rb = (wmean(pm[both], w[both]) / rp[both].mean()) if (pm_cov >= 0.5 and both.mean() >= 0.5) else np.nan
        t = g.temp.values
        out.append(dict(dT=dT, ref_cov_t=ref_cov_t,
                        drop_pre=pre.temp.median() - np.nanmedian(t[3:]),
                        trend=np.nanmedian(t[-5:]) - np.nanmedian(t[3:6]),
                        temp_min=np.nanmin(t[3:]) if np.isfinite(t[3:]).any() else np.nan,
                        ratio_a=stop_pm / pre_pm if pre_pm and pre_pm > 0 else np.nan,
                        ratio_b=rb, pre_pm=pre_pm))
    return pd.concat([st.reset_index(drop=True), pd.DataFrame(out)], axis=1)


def classify(st, kind="primary"):
    rel = st.dT - st.groupby("person").dT.transform("median")
    st = st.assign(dT_rel=rel)
    if kind == "primary":
        typ = np.where(rel <= TH_REF, "refuge", np.where(rel >= TH_OUT, "outdoor", "ambiguous"))
    elif kind == "th-2":
        typ = np.where(rel <= -2, "refuge", np.where(rel >= TH_OUT, "outdoor", "ambiguous"))
    elif kind == "th-4":
        typ = np.where(rel <= -4, "refuge", np.where(rel >= TH_OUT, "outdoor", "ambiguous"))
    elif kind == "brief_drop2":
        typ = np.where(st.drop_pre >= 2, "refuge", np.where(st.drop_pre < 0.5, "outdoor", "ambiguous"))
    elif kind == "uncentred_dT-2":
        typ = np.where(st.dT <= -2, "refuge", np.where(st.dT >= 0, "outdoor", "ambiguous"))
    typ = np.where(st.dT.isna() & (kind != "brief_drop2"), "unclassified", typ)
    if kind == "brief_drop2":
        typ = np.where(st.drop_pre.isna(), "unclassified", typ)
    return st.assign(type=typ)


def per_rider_C(st, col, min_n=3):
    rows = []
    for pid, g in st.groupby("person"):
        a = np.log(g.loc[(g.type == "refuge") & (g[col] > 0), col])
        b = np.log(g.loc[(g.type == "outdoor") & (g[col] > 0), col])
        if len(a) >= min_n and len(b) >= min_n:
            rows.append(dict(person=pid, logC=a.mean() - b.mean(), n_ref=len(a), n_out=len(b),
                             gm_ref=np.exp(a.mean()), gm_out=np.exp(b.mean())))
    return pd.DataFrame(rows)


def pooled(pr, boot=True):
    if len(pr) == 0:
        return np.nan, np.nan, np.nan, None
    x = pr.logC.values
    est = np.exp(x.mean())
    if not boot:
        return est, np.nan, np.nan, None
    rng = np.random.default_rng(SEED)
    bs = np.exp(np.array([x[rng.integers(0, len(x), len(x))].mean() for _ in range(NBOOT)]))
    return est, np.percentile(bs, 2.5), np.percentile(bs, 97.5), bs


def averted(st, D, ons, Cmap, Cpool, mode="C", shift=0.0, mask=None):
    """Returns per-person dataframe: on-shift minutes, refuge minutes, actual & cf means, averted %."""
    rows = []
    ref_rows = {}
    for r in st[st.type == "refuge"].itertuples():
        for k in range(r.r0, r.r1 + 1):
            ref_rows[(r.node, k)] = r
    for (n, pid), g in ons.groupby(["node", "person"]):
        d = D[n].loc[g.row.values]
        if mask is not None:
            d = d[mask(d)]
        if len(d) == 0:
            continue
        pm = d.pm.values + shift
        w = d.n_pm.fillna(0).values
        cf = pm.copy()
        isref = np.array([(n, k) in ref_rows for k in d.index])
        C = Cmap.get(pid, Cpool)
        if mode == "C":
            cf[isref] = pm[isref] / C
        elif mode == "ref_raw":
            rp = d.ref_pm.values + shift
            sel = isref & np.isfinite(rp)
            cf[sel] = rp[sel]
        elif mode == "own_pre":
            pre = np.array([ref_rows[(n, k)].pre_pm + shift if (n, k) in ref_rows else np.nan for k in d.index])
            sel = isref & np.isfinite(pre)
            cf[sel] = pre[sel]
        rows.append(dict(node=n, person=pid, city=d.city.mode().iloc[0], on_min=len(d), ref_min=int(isref.sum()),
                         actual=wmean(pm, w), cf=wmean(cf, w)))
    a = pd.DataFrame(rows)
    a = a.groupby(["person"]).apply(lambda g: pd.Series(dict(
        city=g.city.iloc[0], on_min=g.on_min.sum(), ref_min=g.ref_min.sum(),
        actual=np.average(g.actual, weights=g.on_min), cf=np.average(g.cf, weights=g.on_min))),
        include_groups=False).reset_index()
    a["ref_share"] = a.ref_min / a.on_min
    a["averted"] = 1 - a.actual / a.cf
    return a


def summarize_averted(a):
    """Equal-weight mean of per-rider averted %, and the pooled (sum) version."""
    return a.averted.mean(), 1 - a.actual.mul(a.on_min).sum() / a.cf.mul(a.on_min).sum()


def svg_forest(pr, est, lo, hi, path):
    pr = pr.sort_values("logC").reset_index(drop=True)
    n = len(pr); H = 40 + 18 * (n + 2); Wd = 560
    x0, x1 = 140, 520
    lo_v, hi_v = np.log(0.3), np.log(2.0)
    X = lambda v: x0 + (np.log(v) - lo_v) / (hi_v - lo_v) * (x1 - x0)
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{Wd}" height="{H}" font-family="sans-serif" font-size="11">',
         '<rect width="100%" height="100%" fill="white"/>',
         f'<text x="10" y="16" font-size="12" font-weight="bold">Refuge-type vs outdoor stops: PM ratio (concurrent-reference), per rider</text>']
    for v in [0.3, 0.5, 0.7, 1.0, 1.5, 2.0]:
        s.append(f'<line x1="{X(v):.1f}" y1="28" x2="{X(v):.1f}" y2="{H-20}" stroke="{"#444" if v == 1 else "#ddd"}"/>')
        s.append(f'<text x="{X(v):.1f}" y="{H-6}" text-anchor="middle">{v}</text>')
    for i, r in pr.iterrows():
        y = 40 + 18 * i
        c = np.exp(r.logC)
        s.append(f'<text x="10" y="{y+4}">{r.person} ({int(r.n_ref)}/{int(r.n_out)})</text>')
        s.append(f'<circle cx="{X(min(max(c,0.3),2.0)):.1f}" cy="{y}" r="4" fill="#3b6ea5"/>')
    y = 40 + 18 * (n + 0.8)
    s.append(f'<text x="10" y="{y+4}" font-weight="bold">Pooled</text>')
    s.append(f'<line x1="{X(lo):.1f}" y1="{y}" x2="{X(hi):.1f}" y2="{y}" stroke="#c0392b" stroke-width="2"/>')
    s.append(f'<rect x="{X(est)-5:.1f}" y="{y-5}" width="10" height="10" fill="#c0392b"/>')
    s.append('</svg>')
    open(path, "w").write("\n".join(s))


def svg_hist(st, path):
    v = st.dT_rel.dropna().clip(-12, 8)
    bins = np.arange(-12, 8.5, 0.5)
    h, e = np.histogram(v, bins)
    Wd, H = 560, 260; x0, x1, y0, y1 = 50, 540, 220, 30
    X = lambda t: x0 + (t + 12) / 20 * (x1 - x0)
    Y = lambda c: y0 - c / h.max() * (y0 - y1)
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{Wd}" height="{H}" font-family="sans-serif" font-size="11">',
         '<rect width="100%" height="100%" fill="white"/>',
         '<text x="10" y="16" font-size="12" font-weight="bold">Stop device temperature vs concurrent road reference, rider-centred (°C)</text>']
    for a, c in zip(e[:-1], h):
        col = "#3b6ea5" if a + 0.5 <= TH_REF else ("#e67e22" if a >= TH_OUT else "#bbb")
        s.append(f'<rect x="{X(a):.1f}" y="{Y(c):.1f}" width="{X(a+0.5)-X(a)-1:.1f}" height="{y0-Y(c):.1f}" fill="{col}"/>')
    for t in range(-12, 9, 2):
        s.append(f'<text x="{X(t):.1f}" y="{y0+14}" text-anchor="middle">{t}</text>')
    s.append(f'<text x="{X(-11):.1f}" y="45" fill="#3b6ea5">refuge-type ≤ {TH_REF}</text>')
    s.append(f'<text x="{X(3):.1f}" y="45" fill="#e67e22">outdoor ≥ {TH_OUT}</text>')
    s.append(f'<text x="{X(-2):.1f}" y="{y0+30}" text-anchor="middle">n stops = {len(v)}; grey = ambiguous</text>')
    s.append('</svg>')
    open(path, "w").write("\n".join(s))


def main():
    global COH
    ap = argparse.ArgumentParser()
    ap.add_argument("--pm-col", default="pm")
    ap.add_argument("--out", default=os.path.join(ROOT, "refuge"))
    A = ap.parse_args()
    T = os.path.join(A.out, "tables"); F = os.path.join(A.out, "figures")
    os.makedirs(T, exist_ok=True); os.makedirs(F, exist_ok=True)
    COH = pd.read_csv(os.path.join(ROOT, "foundation", "cohort.csv"), dtype={"sensor": str})
    COH = COH[COH.role == "rider"].set_index("pid")
    COH["enrol_start"] = pd.to_datetime(COH.enrol_start); COH["enrol_end"] = pd.to_datetime(COH.enrol_end)

    D = references(load(A.pm_col))
    st0, ons = find_stops(D)
    st = classify(stop_metrics(st0, D))
    st["period"] = np.where(pd.to_datetime(st.wd).dt.month.isin([3, 4]), "Mar-Apr", "Dec-Feb")
    st.drop(columns=["r0", "r1"]).to_csv(os.path.join(T, "stops.csv"), index=False)
    L = []  # report lines
    p = lambda *x: L.append(" ".join(str(y) for y in x))

    # feasibility
    p("## Feasibility")
    p(f"riders with on-shift data: {ons.person.nunique()}; on-shift minutes: {len(ons)}")
    p(f"stops >=10 min, >300 m from home: {len(st)}; classifiable (temp ref >=50%): {st.dT.notna().sum()}")
    p("types:", st.type.value_counts().to_dict())
    p(f"with ratio_b: {st.ratio_b.notna().sum()}; with ratio_a: {st.ratio_a.notna().sum()}")
    p("by type x ratio_b available:", st[st.ratio_b.notna()].type.value_counts().to_dict())

    # primary
    res = {}
    for col in ["ratio_b", "ratio_a"]:
        pr = per_rider_C(st, col)
        est, lo, hi, bs = pooled(pr)
        res[col] = (pr, est, lo, hi, bs)
        p(f"\n## Contrast {col}: riders k={len(pr)}; C={est:.3f} [{lo:.3f}, {hi:.3f}] -> reduction {100*(1-est):.1f}% [{100*(1-hi):.1f}, {100*(1-lo):.1f}]")
        p(f"  per-rider C range {np.exp(pr.logC.min()):.2f}-{np.exp(pr.logC.max()):.2f}; riders with C<1: {(pr.logC<0).sum()}/{len(pr)}")
        p(f"  GM ratio refuge stops (median over riders) {pr.gm_ref.median():.3f}; outdoor {pr.gm_out.median():.3f}")
        pr.assign(C=np.exp(pr.logC)).to_csv(os.path.join(T, f"per_rider_contrast_{col}.csv"), index=False)
    prb, Cb, lob, hib, bsb = res["ratio_b"]
    Cmap = dict(zip(prb.person, np.exp(prb.logC)))

    # all stops vs road
    g = st[st.ratio_b > 0].groupby("person").ratio_b.apply(lambda x: np.exp(np.log(x).mean()))
    p(f"\n## All stops vs moving reference (descriptive): median over riders of GM ratio_b = {g.median():.3f} (k={len(g)})")

    # averted
    p("\n## Exposure averted (on-shift)")
    av = averted(st, D, ons, Cmap, Cb)
    av["C_used"] = av.person.map(Cmap).fillna(Cb)
    av.to_csv(os.path.join(T, "averted_per_rider.csv"), index=False)
    m_eq, m_pool = summarize_averted(av)
    p(f"refuge time share: median over riders {100*av.ref_share.median():.1f}% (IQR {100*av.ref_share.quantile(.25):.1f}-{100*av.ref_share.quantile(.75):.1f}); pooled {100*av.ref_min.sum()/av.on_min.sum():.1f}%")
    for c, gg in av.groupby("city"):
        p(f"  {c}: share median {100*gg.ref_share.median():.1f}%, pooled {100*gg.ref_min.sum()/gg.on_min.sum():.1f}%; averted mean {100*gg.averted.mean():.2f}%")
    p(f"averted (primary, per-rider C): mean over riders {100*m_eq:.2f}%, pooled {100*m_pool:.2f}%")
    # all stop-type time shares
    sh = st.groupby(["person", "type"]).L.sum().unstack(fill_value=0)
    sh = sh.div(ons.groupby("person").size(), axis=0)
    sh.to_csv(os.path.join(T, "stop_time_shares.csv"))
    p("median share of on-shift time by stop type:", (100 * sh.median()).round(1).to_dict())
    sens = []
    for mode in ["ref_raw", "own_pre"]:
        a2 = averted(st, D, ons, Cmap, Cb, mode=mode)
        e1, e2 = summarize_averted(a2); sens.append((f"cf={mode}", e1, e2))
    a_up = averted(st, D, ons, {}, lob)  # everyone at the protective CI bound
    e1, e2 = summarize_averted(a_up); sens.append((f"all riders C={lob:.3f} (CI bound)", e1, e2))
    a_pc = averted(st, D, ons, {}, Cb)
    e1, e2 = summarize_averted(a_pc); sens.append((f"all riders pooled C={Cb:.3f}", e1, e2))
    for s_, e1, e2 in sens:
        p(f"  sensitivity {s_}: mean {100*e1:.2f}%, pooled {100*e2:.2f}%")

    # strata
    p("\n## City x period (descriptive)")
    strat = []
    for (c, per), gs in st.groupby(["city", "period"]):
        pr = per_rider_C(gs, "ratio_b", min_n=2)
        est = pooled(pr, boot=False)[0]
        months = [3, 4] if per == "Mar-Apr" else [12, 1, 2]
        a = averted(gs, D, ons, Cmap, Cb, mask=lambda d, c=c, m=months: (d.city == c) & d.wd.dt.month.isin(m))
        strat.append(dict(city=c, period=per, k_contrast=len(pr), C=est, reduction=1 - est,
                          n_ref=(gs.type == "refuge").sum(), n_out=(gs.type == "outdoor").sum(),
                          riders=len(a), ref_share_pooled=a.ref_min.sum() / a.on_min.sum(),
                          averted_mean=a.averted.mean()))
    strat = pd.DataFrame(strat); strat.to_csv(os.path.join(T, "strata_city_period.csv"), index=False)
    p(strat.round(3).to_string(index=False))

    # marker sensitivities
    p("\n## Marker sensitivities (ratio_b)")
    ms = []
    for kind in ["primary", "th-2", "th-4", "brief_drop2", "uncentred_dT-2"]:
        s2 = classify(st.drop(columns=["type", "dT_rel"]), kind)
        pr = per_rider_C(s2, "ratio_b")
        est, lo, hi, _ = pooled(pr)
        cm = dict(zip(pr.person, np.exp(pr.logC)))
        a = averted(s2, D, ons, cm, est if np.isfinite(est) else 1.0)
        ms.append(dict(marker=kind, n_ref=(s2.type == "refuge").sum(), n_out=(s2.type == "outdoor").sum(),
                       k=len(pr), C=est, lo=lo, hi=hi, ref_share_pooled=a.ref_min.sum() / a.on_min.sum(),
                       averted_mean=a.averted.mean()))
    ms = pd.DataFrame(ms); ms.to_csv(os.path.join(T, "marker_sensitivity.csv"), index=False)
    p(ms.round(3).to_string(index=False))

    # intercept
    p("\n## Intercept sensitivity (ratio_b)")
    for sh_ in [-5.0, 5.0]:
        s3 = stop_metrics(st0, D, shift=sh_)
        s3["type"] = st.type.values
        pr = per_rider_C(s3, "ratio_b"); est, lo, hi, _ = pooled(pr)
        a = averted(s3.assign(pre_pm=s3.pre_pm), D, ons, dict(zip(pr.person, np.exp(pr.logC))), est, shift=sh_)
        p(f"  shift {sh_:+}: C={est:.3f} [{lo:.3f}, {hi:.3f}]; averted mean {100*a.averted.mean():.2f}%")

    # placement checks
    p("\n## Sensor placement / plausibility")
    rf = st[st.type == "refuge"]; od = st[st.type == "outdoor"]
    p(f"refuge stops with min device temp < 30 C: {(rf.temp_min < 30).mean():.3f}; outdoor: {(od.temp_min < 30).mean():.3f}")
    p(f"median device-temp min in refuge stops {rf.temp_min.median():.1f} C, outdoor {od.temp_min.median():.1f} C")
    p(f"no-GPS minute share: refuge {rf.nogps.sum()/rf.L.sum():.3f}, outdoor {od.nogps.sum()/od.L.sum():.3f}")
    p(f"median refuge stop length {rf.L.median():.0f} min, outdoor {od.L.median():.0f}")
    p(f"refuge-stop local-hour distribution: {pd.to_datetime(rf.start+7*3600,unit='s').dt.hour.value_counts().sort_index().to_dict()}")

    # joint display
    jd = []
    qm = pd.read_csv(os.path.join(ROOT, "qual", "03_rider_code_matrix_v1.csv"), encoding="utf-8-sig")
    for P_, R_ in P2R.items():
        q = qm.loc[qm.P == P_, "REFUGE"]
        a = av[av.person == R_]; c = prb[prb.person == R_]
        s_ = st[st.person == R_]
        jd.append(dict(P=P_, R=R_, city=a.city.iloc[0] if len(a) else "", said_refuge=q.iloc[0] if len(q) and q.iloc[0] == q.iloc[0] else "",
                       n_stops=len(s_), n_refuge=(s_.type == "refuge").sum(), n_outdoor=(s_.type == "outdoor").sum(),
                       refuge_share=a.ref_share.iloc[0] if len(a) else np.nan,
                       C=np.exp(c.logC.iloc[0]) if len(c) else np.nan,
                       refuge_min_temp_lt30=(s_[s_.type == "refuge"].temp_min < 30).mean() if (s_.type == "refuge").any() else np.nan))
    jd = pd.DataFrame(jd); jd.to_csv(os.path.join(T, "joint_display.csv"), index=False)
    p("\n## Joint display"); p(jd.round(3).to_string(index=False))

    svg_forest(prb, Cb, lob, hib, os.path.join(F, "fig1_refuge_contrast_per_rider.svg"))
    svg_hist(st, os.path.join(F, "fig2_stop_temperature_marker.svg"))
    open(os.path.join(T, "run_log.txt"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
