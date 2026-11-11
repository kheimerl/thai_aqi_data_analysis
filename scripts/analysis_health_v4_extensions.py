#!/usr/bin/env python3
"""Extensions X1-X7 (see analysis_v3/SPEC_EXTENSIONS.md). Imports the statistics helpers from analysis_health_v3.py.
All clocks: Asia/Bangkok explicit (OFFSET = 7 h). Never use the wrong-clock build.

Usage:
  python3 -I scripts/analysis_health_v4_extensions.py --data-dir data --nodes-dir PATH/aq_by_node --out analysis_v3/extensions
"""
import argparse, ast, importlib.util, math, os, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("v3", os.path.join(HERE, "analysis_health_v3.py"))
v3 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(v3)
OFFSET, EPOCH = v3.OFFSET, v3.EPOCH
STAFF_NOT_OFFICE = {"Rach", "Rac", "test", "Patte", "Nussara"}  # excluded from this file entirely
OFFICE_PVT = {"adisorn": "Grab-6010", "Preechai": "Grab-6009"}
THAI_24H = 37.5
NMODELS = [0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True); ap.add_argument("--nodes-dir", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    out = []
    def P(s=""):
        print(s); out.append(s)

    # ---------- roster, nodes ----------
    roster = pd.read_excel(os.path.join(a.data_dir, "2025-2026 Rider-Sensor-Summary.xlsx"), sheet_name="summary")
    roster["u"] = roster["PVT user"].astype(str).str.strip().str.lower()
    roster["node"] = "Grab-" + roster["Sensor no."].astype(str).str.replace(r"\.0$", "", regex=True)
    roster["sd"] = roster["Start date"].apply(v3.xl_date); roster["ed"] = roster["End date"].apply(v3.xl_date)
    roster["city"] = roster["Region"].astype(str)
    def resolve(u, d):
        for r in roster[roster.u == u.lower()].itertuples():
            if r.sd <= d <= r.ed: return r.node, r.city
        return None, None
    nodes = {}
    def node(n):
        if n not in nodes:
            p = os.path.join(a.nodes_dir, f"{n}.csv"); nodes[n] = v3.Node(p) if os.path.exists(p) else None
        return nodes[n]

    # ---------- hourly background / local decomposition ----------
    rows = []
    for nname, grp in roster.groupby("node"):
        nd = node(nname)
        if nd is None: continue
        ok = ~np.isnan(nd.pmv); hr = (nd.ts[ok].astype(np.int64) + OFFSET) // 3600
        s = pd.DataFrame({"hr": hr, "pm": nd.pmv[ok]}).groupby("hr").pm.agg(["mean", "count"]); s = s[s["count"] >= 600]
        dates = pd.to_datetime(s.index.to_numpy() * 3600, unit="s").date
        keep = np.array([any(r.sd <= d <= r.ed for r in grp.itertuples()) for d in dates])
        s = s[keep]; rows.append(pd.DataFrame({"node": nname, "city": grp.city.iloc[0], "hr": s.index.to_numpy(), "pm": s["mean"].to_numpy()}))
    hh = pd.concat(rows); g = hh.groupby(["city", "hr"]).pm.agg(["sum", "count"]).reset_index()
    hh = hh.merge(g, on=["city", "hr"]); hh = hh[hh["count"] >= 2].copy()
    hh["bg"] = (hh["sum"] - hh.pm) / (hh["count"] - 1); hh["loc"] = hh.pm - hh.bg
    HB = {n: (q.hr.to_numpy(), q.bg.to_numpy(), q["loc"].to_numpy()) for n, q in hh.sort_values("hr").groupby("node")}
    P(f"# Extensions v4 (Asia/Bangkok explicit). Rider-hours with a background from >=1 other rider: {len(hh)}; "
      f"mean local increment {hh['loc'].mean():+.1f} µg/m³; share of rider-hour PM variance from background {100*hh.bg.var()/hh.pm.var():.0f}% (rough)")
    def decomp(n, t0, t1):
        if n not in HB: return np.nan, np.nan
        hrs, bg, lo = HB[n]; h0 = (int(t0) + OFFSET) // 3600; h1 = (int(t1) + OFFSET) // 3600 - 1
        i0 = np.searchsorted(hrs, h0, "left"); i1 = np.searchsorted(hrs, h1, "right")
        return (bg[i0:i1].mean(), lo[i0:i1].mean()) if i1 - i0 >= 2 else (np.nan, np.nan)

    # ---------- PVT sessions (riders + 2 office workers) ----------
    pvt = pd.read_csv(os.path.join(a.data_dir, "pvt.csv"), usecols=["timestamp", "username", "response_times"])
    pvt["username"] = pvt.username.astype(str).str.strip(); pvt = pvt[~pvt.username.isin(STAFF_NOT_OFFICE)].copy()
    pvt["naive"] = pd.to_datetime(pvt.timestamp); pvt = pvt.sort_values(["username", "naive"]).reset_index(drop=True)
    def parse(s):
        try: return np.array(ast.literal_eval(s), float)
        except Exception: return np.array([])
    pvt["rts"] = pvt.response_times.apply(parse)
    med = pvt.groupby("username").rts.apply(lambda s: np.median(np.concatenate([x[x >= 100] for x in s])))
    def feats(r):
        v = r.rts[r.rts >= 100]; vp = r.rts[r.rts > 0]; n = len(v); thr = 2 * med[r.username]
        if n < 30: return pd.Series(dict(y1=np.nan, y2=np.nan, y3=np.nan, y6=np.nan, y7=np.nan, y8=np.nan))
        sl = np.sort(v)[::-1][: max(1, int(round(0.1 * n)))]; k = n // 3
        return pd.Series(dict(y1=np.mean(1000 / vp), y2=100 * np.mean(v >= thr), y3=np.mean(1000 / sl),
                              y6=np.mean(1000 / v[-k:]) - np.mean(1000 / v[:k]), y7=np.std(np.log(v), ddof=1),
                              y8=np.percentile(v, 90) / np.median(v)))
    pvt = pd.concat([pvt.drop(columns=["rts", "response_times"]).assign(rts=None) if False else pvt, pvt.apply(feats, axis=1)], axis=1)
    pvt["session_number"] = pvt.groupby("username").cumcount() + 1
    pvt["hour_of_day"] = pvt.naive.dt.hour + pvt.naive.dt.minute / 60
    pvt["date"] = pvt.naive.dt.date; pvt["ts"] = v3.to_unix(pvt.naive)
    pvt["office"] = pvt.username.isin(OFFICE_PVT)
    nn = [OFFICE_PVT.get(u) and (OFFICE_PVT[u], None) or resolve(u, d) for u, d in zip(pvt.username, pvt.date)]
    pvt["node"] = [x[0] for x in nn]; pvt["city"] = [x[1] for x in nn]
    for c in ["pm", "temp", "rh", "shift", "B", "L"]: pvt[c] = np.nan
    for i, r in pvt.iterrows():
        nd = node(r.node) if r.node else None
        if nd is None: continue
        d0 = v3.day_start_unix(r.date); m, n, first = nd.win(d0, r.ts)
        if n >= 60:
            pvt.at[i, "pm"] = m; pvt.at[i, "temp"] = nd.win(d0, r.ts, "temp")[0]; pvt.at[i, "rh"] = nd.win(d0, r.ts, "rh")[0]; pvt.at[i, "shift"] = (r.ts - first) / 60
            pvt.at[i, "B"], pvt.at[i, "L"] = decomp(r.node, d0, r.ts)
    riders = pvt[~pvt.office].copy(); office = pvt[pvt.office].copy()
    ctrl = ["temp", "rh", "shift", "session_number", "hour_of_day"]
    riders["hi37"] = np.where(riders.pm.isna(), np.nan, (riders.pm >= THAI_24H).astype(float))
    def top_bottom(df, col):
        df = df.copy(); df["tb"] = np.nan
        for u, g in df.groupby("username"):
            x = g[col].dropna()
            if len(x) < 30: continue
            q1, q2 = x.quantile([1 / 3, 2 / 3]); idx = g.index
            df.loc[idx[g[col] <= q1], "tb"] = 0.0; df.loc[idx[g[col] >= q2], "tb"] = 1.0
        return df
    riders = top_bottom(riders, "pm")
    P(f"Riders {riders.username.nunique()}, sessions {len(riders)}, with same-day PM {riders.pm.notna().sum()}, with B/L decomposition {riders.B.notna().sum()}; "
      f"high-day (>= {THAI_24H}) share {100*riders.hi37.mean():.1f}%")

    # ---------- shared model runner ----------
    def run(df, y, x, controls, label, groups="date", scale=10, min_n=15, harm=-1, min_cat=0):
        res = []
        for u, g in df.groupby("username"):
            if min_cat:
                c = g.dropna(subset=[x, y])[x].value_counts()
                if len(c) < 2 or c.min() < min_cat: continue
            f = v3.rider_fit(g, y, x, controls, groups, min_n)
            if f: res.append(f)
        NMODELS[0] += 1
        if len(res) < 5: P(f"| {label} | too few riders ({len(res)}) |"); return None
        b = np.array([r["beta"] for r in res]); s = np.array([r["se"] for r in res])
        pl = v3.pool(b, s); o = dict(label=label, **pl, n=sum(r["n"] for r in res))
        bz = np.array([r["beta"] / r["sd_y"] for r in res]) * scale; sz = np.array([r["se"] / r["sd_y"] for r in res]) * scale
        bb = v3.bayes(bz, sz); o["p_harm"] = bb["p_harm"] if harm < 0 else 1 - bb["p_harm"]
        o["scale"] = scale; o["harm"] = harm; return o
    HDR = "| Model | riders / n | effect (HKSJ pooled) | 95% CI (HKSJ) | p HKSJ | p DL | I² % | riders in harm direction | P(harm) Bayes |\n|---|---|---|---|---|---|---|---|---|"
    def row(o):
        if not o: return
        sc = o["scale"]; harmshare = (o["share_neg"] if o["harm"] < 0 else 1 - o["share_neg"]) * 100
        P(f"| {o['label']} | {o['k']} / {o['n']} | {o['mu']*sc:+.4f} | [{o['lo_hk']*sc:+.4f}, {o['hi_hk']*sc:+.4f}] | {o['p_hk']:.3f} | {o['p_dl']:.3f} | {o['i2']:.0f} | {harmshare:.0f}% | {o['p_harm']:.2f} |")

    # ================= X2 bounded null =================
    P("\n## X2. Bounded null for cognition (what the data can rule out), same-day cumulative PM, +60 µg/m³ (about 20 → 80)")
    P(HDR)
    base = {}
    for y, lab, h in [("y1", "Y1 mean speed (Hz)", -1), ("y3", "Y3 slowest-10% speed (Hz)", -1), ("y2", "Y2 lapse % (≥2× own median RT)", 1)]:
        base[y] = run(riders, y, "pm", ctrl, lab, harm=h); row(base[y])
    m = riders.y1.mean(); sdw = riders.groupby("username").y1.std().mean()
    o = base["y1"]; P(f"\nMean speed {m:.2f} Hz (RT ≈ {1000/m:.0f} ms); average within-rider SD of session speed {sdw:.2f} Hz.")
    for nm, val in [("point estimate", o["mu"]), ("95% CI slowing bound (lower)", o["lo_hk"]), ("95% CI benefit bound (upper)", o["hi_hk"])]:
        d = val * 60; P(f"- {nm}: {d:+.3f} Hz = {100*d/m:+.1f}% of mean speed = RT {1000/(m+d)-1000/m:+.1f} ms = {d/sdw:+.2f} within-rider SD, per +60 µg/m³")
    o = base["y2"]; P(f"- Lapse % (rider-relative): point {o['mu']*60:+.2f}, 95% CI [{o['lo_hk']*60:+.2f}, {o['hi_hk']*60:+.2f}] percentage points per +60 µg/m³ (baseline lapse share ≈ {riders.y2.mean():.1f}%)")

    # ================= X1 background vs local =================
    P("\n## X1. Regional background (B) vs local increment (L), same-day cumulative, per +10 µg/m³, each adjusted for the other")
    P(HDR)
    for y, lab, h in [("y1", "Y1 speed", -1), ("y3", "Y3 slowest-10%", -1), ("y2", "Y2 lapse %", 1)]:
        row(run(riders, y, "B", ctrl + ["L"], f"{lab}: background B | L", harm=h))
        row(run(riders, y, "L", ctrl + ["B"], f"{lab}: local increment L | B", harm=h))

    # ================= X3 negative control =================
    P("\n## X3. Negative-control population: office workers (staff) PVT vs same-day PM2.5 (per +10 µg/m³)")
    for u, g in office.groupby("username"):
        d = g.dropna(subset=["y1", "pm"] + ctrl)
        P(f"- {u}: sessions {len(g)}, with exposure {len(d)}, days {d.date.nunique()}, mean PM {d.pm.mean():.1f} (SD {d.pm.std():.1f}); rider same-day PM mean {riders.pm.mean():.1f} (SD {riders.pm.std():.1f})")
        f = v3.rider_fit(g, "y1", "pm", ctrl, "date", 15)
        if f: P(f"  slope {f['beta']*10:+.4f} Hz per +10 µg/m³ (SE {f['se']*10:.4f}, n {f['n']})  [compare riders pooled in X2; single-person estimate, descriptive only]")
        else: P("  too few sessions for a stable single-person slope; counts only")

    # ================= X4 other PVT outcomes =================
    P("\n## X4. Other PVT outcomes (exploratory family, Holm across the three)")
    P(HDR)
    fam = [run(riders, "y6", "pm", ctrl, "Y6 within-test decrement (last − first third speed, Hz; lower = worse)", harm=-1),
           run(riders, "y7", "pm", ctrl, "Y7 variability (SD of log RT; higher = worse)", harm=1),
           run(riders, "y8", "pm", ctrl, "Y8 slow-tail ratio (P90 / median RT; higher = worse)", harm=1)]
    for o in fam: row(o)
    pv = [o["p_hk"] for o in fam if o]; P("Holm-adjusted HKSJ p: " + ", ".join(f"{x:.3f}" for x in v3.holm(np.array(pv))))

    # ================= X5 thresholds / tertiles (PVT) =================
    P("\n## X5a. High-exposure days vs others and within-rider top vs bottom tertile (PVT; effect = difference in outcome, not per 10 µg)")
    P(HDR)
    for y, lab, h in [("y1", "Y1 speed (Hz)", -1), ("y3", "Y3 slowest-10% (Hz)", -1), ("y2", "Y2 lapse %", 1)]:
        row(run(riders, y, "hi37", ctrl, f"{lab}: same-day mean ≥ {THAI_24H} vs below", scale=1, harm=h, min_cat=5))
        row(run(riders, y, "tb", ctrl, f"{lab}: top vs bottom own-tertile of same-day PM", scale=1, harm=h, min_cat=5))

    # ================= Peak flow =================
    P("\n## X6. Peak flow (evening, % of personal best) robustness")
    P(HDR)
    pef = pd.read_csv(os.path.join(a.data_dir, "peakflow.csv")); pef["username"] = pef.username.astype(str).str.strip()
    pef = pef[~pef.username.isin(v3.STAFF)].copy(); pef["naive"] = pd.to_datetime(pef.timestamp)
    pef = pef.drop_duplicates(["username", "naive"]); pef = pef[pef.peak_flow >= 50].copy()
    pef["pct"] = 100 * pef.peak_flow / pef.username.map(pef.groupby("username").peak_flow.quantile(0.95))
    pef["pct90"] = 100 * pef.peak_flow / pef.username.map(pef.groupby("username").peak_flow.quantile(0.90))
    pef["workday"] = (pef.naive - pd.Timedelta(hours=4)).dt.date; h = pef.naive.dt.hour
    pef["slot"] = np.where((h >= 4) & (h < 13), "morning", np.where((h >= 15) | (h < 4), "evening", "other")); pef["ts"] = v3.to_unix(pef.naive)
    ev = pef[pef.slot == "evening"].sort_values("naive").groupby(["username", "workday"]).last().reset_index()
    t0 = pef.groupby("username").naive.min().dt.normalize()
    for c in ["pm", "temp", "rh", "shift", "pm4h", "B", "L"]: ev[c] = np.nan
    ev["city"] = None
    for i, r in ev.iterrows():
        nr = resolve(r.username, r.workday); ev.at[i, "city"] = nr[1]; nd = node(nr[0]) if nr[0] else None
        if nd is None: continue
        d0 = v3.day_start_unix(r.workday); m_, n_, first = nd.win(d0, r.ts)
        if n_ >= 60:
            ev.at[i, "pm"] = m_; ev.at[i, "temp"] = nd.win(d0, r.ts, "temp")[0]; ev.at[i, "rh"] = nd.win(d0, r.ts, "rh")[0]; ev.at[i, "shift"] = (r.ts - first) / 60
            ev.at[i, "B"], ev.at[i, "L"] = decomp(nr[0], d0, r.ts)
        m4, n4, _ = nd.win(r.ts - 4 * 3600, r.ts)
        if n4 >= 60: ev.at[i, "pm4h"] = m4
    ev["studyday"] = (ev.naive - ev.username.map(t0)).dt.days; ev["date"] = ev.workday; ev["hour_read"] = ev.naive.dt.hour + ev.naive.dt.minute / 60
    ev["hour_read"] = np.where(ev.hour_read < 4, ev.hour_read + 24, ev.hour_read)
    ev["hi37"] = np.where(ev.pm.isna(), np.nan, (ev.pm >= THAI_24H).astype(float)); ev = top_bottom(ev, "pm")
    base_c = ["temp", "rh", "studyday"]; mn = 12
    P(f"evening readings {len(ev)}, with same-day PM {ev.pm.notna().sum()}, with B/L {ev.B.notna().sum()}")
    row(run(ev, "pct", "pm", base_c, "PEF reference: same-day cumulative PM", groups=None, min_n=mn))
    row(run(ev, "pct", "pm", base_c + ["hour_read"], "+ clock hour of the reading", groups=None, min_n=mn))
    row(run(ev, "pct90", "pm", base_c, "personal best = 90th percentile", groups=None, min_n=mn))
    row(run(ev, "pct", "pm4h", base_c, "preceding 4 h exposure", groups=None, min_n=mn))
    row(run(ev, "pct", "B", base_c + ["L"], "background B | L", groups=None, min_n=mn))
    row(run(ev, "pct", "L", base_c + ["B"], "local increment L | B", groups=None, min_n=mn))
    row(run(ev, "pct", "hi37", base_c, f"day mean ≥ {THAI_24H} vs below (diff in %PB)", groups=None, min_n=mn, scale=1, min_cat=4))
    row(run(ev, "pct", "tb", base_c, "top vs bottom own-tertile (diff in %PB)", groups=None, min_n=mn, scale=1, min_cat=4))

    # ================= X7 co-movement =================
    P("\n## X7. Co-movement of cognition and lungs (rider-day level, exploratory)")
    dm = riders.groupby(["username", "date"]).y1.mean().reset_index().rename(columns={"y1": "speed_day", "date": "workday"})
    cm = ev[["username", "workday", "pct"]].merge(dm, on=["username", "workday"])
    zs, ses = [], []
    for u, g in cm.groupby("username"):
        if len(g) >= 12 and g.pct.std() > 0 and g.speed_day.std() > 0:
            r_ = np.corrcoef(g.pct, g.speed_day)[0, 1]; zs.append(np.arctanh(r_)); ses.append(1 / math.sqrt(len(g) - 3))
    if len(zs) >= 5:
        pl = v3.pool(zs, ses); NMODELS[0] += 1
        P(f"pooled within-rider correlation between evening %PB and that day's mean PVT speed: r = {math.tanh(pl['mu']):+.3f} "
          f"(HKSJ 95% CI [{math.tanh(pl['lo_hk']):+.3f}, {math.tanh(pl['hi_hk']):+.3f}], p = {pl['p_hk']:.3f}, riders {pl['k']}, rider-days {len(cm)})")

    P(f"\nTotal pooled models run in this file: {NMODELS[0]} (all exploratory; nothing here is a confirmatory test)")
    open(os.path.join(a.out, "results_v4.md"), "w").write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
