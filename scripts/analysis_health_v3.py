#!/usr/bin/env python3
"""Health-outcome analysis v3 (see analysis_v3/SPEC.md). numpy/pandas only (no scipy/statsmodels).

All clocks: PVT and peak-flow timestamps are Asia/Bangkok local; AQ Timestamp is Unix.
Local naive time -> Unix via  - 7 h  (explicit, machine-timezone independent).

Usage:
  python3 -I scripts/analysis_health_v3.py --data-dir data --nodes-dir PATH/aq_by_node --out analysis_v3
"""
import argparse, ast, math, os, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")

OFFSET = 7 * 3600
STAFF = {"Nussara", "Rach", "Rac", "test", "Preechai", "adisorn", "Patte"}
OFFICE = ["adisorn", "Preechai", "Nussara"]
EPOCH = pd.Timestamp("1970-01-01")

# ---------------- statistics helpers ----------------
def betainc(a, b, x):
    if x <= 0: return 0.0
    if x >= 1: return 1.0
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    front = math.exp(lbeta + a * math.log(x) + b * math.log(1 - x))
    def cf(a, b, x):
        tiny = 1e-300; qab = a + b; qap = a + 1; qam = a - 1
        c = 1.0; d = 1 - qab * x / qap
        d = tiny if abs(d) < tiny else d; d = 1 / d; h = d
        for m in range(1, 300):
            m2 = 2 * m
            aa = m * (b - m) * x / ((qam + m2) * (a + m2))
            d = 1 + aa * d; d = tiny if abs(d) < tiny else d
            c = 1 + aa / c; c = tiny if abs(c) < tiny else c
            d = 1 / d; h *= d * c
            aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
            d = 1 + aa * d; d = tiny if abs(d) < tiny else d
            c = 1 + aa / c; c = tiny if abs(c) < tiny else c
            d = 1 / d; de = d * c; h *= de
            if abs(de - 1) < 1e-12: break
        return h
    if x < (a + 1) / (a + b + 2): return front * cf(a, b, x) / a
    return 1 - front * cf(b, a, 1 - x) / b

def p_t(t, df):  # two-sided
    return betainc(df / 2, 0.5, df / (df + t * t))

def p_z(z): return math.erfc(abs(z) / math.sqrt(2))

def t_crit(df, q=0.975):
    lo, hi = 0.0, 50.0
    for _ in range(80):
        mid = (lo + hi) / 2
        cdf = 1 - 0.5 * p_t(mid, df)
        if cdf < q: lo = mid
        else: hi = mid
    return (lo + hi) / 2

def ols(y, X, groups=None, min_g=20):
    n, p = X.shape
    XtXi = np.linalg.pinv(X.T @ X); b = XtXi @ X.T @ y; u = y - X @ b
    g = None if groups is None else np.asarray(groups)
    if g is not None and len(np.unique(g)) >= min_g:
        meat = np.zeros((p, p)); ug = np.unique(g)
        for k in ug:
            m = g == k; s = X[m].T @ u[m]; meat += np.outer(s, s)
        G = len(ug); c = G / (G - 1) * (n - 1) / max(n - p, 1)
    else:
        meat = X.T @ (X * (u ** 2)[:, None]); c = n / max(n - p, 1)
    return b, np.sqrt(np.diag(c * XtXi @ meat @ XtXi)), u

def rider_fit(df, y, x, controls, groups="date", min_n=15):
    cols = [y, x] + controls
    d = df.dropna(subset=cols)
    if len(d) < min_n or d[x].std() < 1e-9: return None
    X = np.column_stack([np.ones(len(d))] + [d[c].to_numpy(float) for c in [x] + controls])
    if np.linalg.matrix_rank(X) < X.shape[1]: return None
    yy = d[y].to_numpy(float)
    b, se, _ = ols(yy, X, d[groups].to_numpy() if groups else None)
    return dict(beta=b[1], se=se[1], n=len(d), sd_y=yy.std(ddof=1))

def pool(betas, ses):
    b = np.asarray(betas, float); s = np.asarray(ses, float); k = len(b)
    w = 1 / s ** 2; mu_f = (w * b).sum() / w.sum(); q = (w * (b - mu_f) ** 2).sum()
    tau2 = max(0.0, (q - (k - 1)) / (w.sum() - (w ** 2).sum() / w.sum()))
    ws = 1 / (s ** 2 + tau2); mu = (ws * b).sum() / ws.sum(); se_dl = math.sqrt(1 / ws.sum())
    z = mu / se_dl
    var_hk = (ws * (b - mu) ** 2).sum() / ((k - 1) * ws.sum())
    se_hk = math.sqrt(max(var_hk, 1 / ws.sum()))  # IntHout floor
    tc = t_crit(k - 1)
    return dict(k=k, mu=mu, se_dl=se_dl, lo_dl=mu - 1.96 * se_dl, hi_dl=mu + 1.96 * se_dl, p_dl=p_z(z),
                se_hk=se_hk, lo_hk=mu - tc * se_hk, hi_hk=mu + tc * se_hk, p_hk=p_t(mu / se_hk, k - 1),
                tau2=tau2, i2=max(0.0, (q - (k - 1)) / q) * 100 if q > 0 else 0.0, share_neg=float((b < 0).mean()))

def bayes(betaz, sez, sd_mu=0.1, sd_tau=0.1):
    b = np.asarray(betaz); s2 = np.asarray(sez) ** 2
    mus = np.linspace(-5 * sd_mu, 5 * sd_mu, 801); taus = np.linspace(0, 5 * sd_tau, 301)
    ll = np.zeros((len(taus), len(mus)))
    for i in range(len(b)):
        v = s2[i] + taus[:, None] ** 2
        ll += -0.5 * np.log(v) - 0.5 * (b[i] - mus[None, :]) ** 2 / v
    lp = ll - 0.5 * (mus[None, :] / sd_mu) ** 2 - 0.5 * (taus[:, None] / sd_tau) ** 2
    pr = np.exp(lp - lp.max()); pr /= pr.sum(); pm = pr.sum(0)
    cdf = np.cumsum(pm)
    return dict(mean=float((pm * mus).sum()), p_harm=float(pm[mus < 0].sum()),
                lo=float(mus[np.searchsorted(cdf, 0.025)]), hi=float(mus[min(np.searchsorted(cdf, 0.975), len(mus) - 1)]))

def holm(ps):
    order = np.argsort(ps); m = len(ps); adj = np.zeros(m); run = 0
    for r, i in enumerate(order):
        run = max(run, (m - r) * ps[i]); adj[i] = min(1.0, run)
    return adj

# ---------------- data ----------------
def xl_date(x):
    if isinstance(x, (int, float, np.integer, np.floating)):
        return (pd.Timestamp("1899-12-30") + pd.Timedelta(days=float(x))).date()
    return pd.Timestamp(x).date()

EMU_TZ = None  # set by --emulate-tz to reproduce a wrong-machine-clock build (what naive datetime.timestamp() does)

def to_unix(naive):  # naive local (Bangkok) -> unix seconds, explicit
    if EMU_TZ:
        loc = naive.dt.tz_localize(EMU_TZ, ambiguous=True, nonexistent="shift_forward")
        return (loc.dt.tz_convert("UTC").dt.tz_localize(None) - EPOCH).dt.total_seconds()
    return (naive - EPOCH).dt.total_seconds() - OFFSET

class Node:
    def __init__(self, path):
        d = pd.read_csv(path, usecols=["ts", "pm", "temp", "rh"], dtype={"ts": "int64"}).sort_values("ts")
        self.ts = d.ts.to_numpy(np.int64).astype(np.int32)
        pm = d.pm.to_numpy(float); pm = np.where((pm < 0) | (pm > 500), np.nan, pm)
        def cs(x):
            ok = ~np.isnan(x)
            return np.concatenate([[0], np.cumsum(np.where(ok, x, 0))]), np.concatenate([[0], np.cumsum(ok)]).astype(np.int32)
        self.c = {"pm": cs(pm), "temp": cs(d.temp.to_numpy(float)), "rh": cs(d.rh.to_numpy(float))}
        self.pmv = pm.astype(np.float32)
    def win(self, t0, t1, key="pm"):
        lo = np.searchsorted(self.ts, t0, "left"); hi = np.searchsorted(self.ts, t1, "right")
        s, n = self.c[key]; cnt = n[hi] - n[lo]
        return (s[hi] - s[lo]) / cnt if cnt > 0 else np.nan, hi - lo, (float(self.ts[lo]) if hi > lo else np.nan)

def day_start_unix(date):
    if EMU_TZ: return pd.Timestamp(date).tz_localize(EMU_TZ).timestamp()
    return (pd.Timestamp(date) - EPOCH).total_seconds() - OFFSET

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True); ap.add_argument("--nodes-dir", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--emulate-tz", default=None, help="DIAGNOSTIC ONLY: treat Bangkok-local PVT/peak-flow times as this tz (e.g. America/Los_Angeles) to reproduce the wrong-clock build")
    a = ap.parse_args()
    global EMU_TZ; EMU_TZ = a.emulate_tz; os.makedirs(a.out, exist_ok=True)
    out = []
    def P(s=""):
        print(s); out.append(s)

    roster = pd.read_excel(os.path.join(a.data_dir, "2025-2026 Rider-Sensor-Summary.xlsx"), sheet_name="summary")
    roster["u"] = roster["PVT user"].astype(str).str.strip().str.lower()
    roster["node"] = "Grab-" + roster["Sensor no."].astype(str).str.replace(r"\.0$", "", regex=True)
    roster["sd"] = roster["Start date"].apply(xl_date); roster["ed"] = roster["End date"].apply(xl_date)
    roster["city"] = roster["Region"].astype(str)
    def resolve(u, d):
        for r in roster[roster.u == u.lower()].itertuples():
            if r.sd <= d <= r.ed: return r.node, r.city
        return None, None
    nodes = {}
    def node(n):
        if n not in nodes:
            p = os.path.join(a.nodes_dir, f"{n}.csv"); nodes[n] = Node(p) if os.path.exists(p) else None
        return nodes[n]

    # ---- PVT ----
    pvt = pd.read_csv(os.path.join(a.data_dir, "pvt.csv"), usecols=["timestamp", "username", "response_times"])
    pvt["username"] = pvt.username.astype(str).str.strip(); pvt["naive"] = pd.to_datetime(pvt.timestamp)
    riders = pvt[~pvt.username.isin(STAFF)].copy().sort_values(["username", "naive"])
    def parse(s):
        try: return np.array(ast.literal_eval(s), float)
        except Exception: return np.array([])
    riders["rts"] = riders.response_times.apply(parse)
    med = riders.groupby("username").rts.apply(lambda s: np.median(np.concatenate([x[x >= 100] for x in s])))
    def feats(r):
        v = r.rts; v = v[v >= 100]; vp = r.rts[r.rts > 0]
        thr = 2 * med[r.username]
        sl = np.sort(v)[::-1][: max(1, int(round(0.1 * len(v))))] if len(v) else np.array([])
        return pd.Series(dict(y1=np.mean(1000 / vp) if len(vp) else np.nan,
                              y2=100 * np.mean(v >= thr) if len(v) else np.nan,
                              y2_500=100 * np.mean(v >= 500) if len(v) else np.nan,
                              y2_1000=100 * np.mean(v >= 1000) if len(v) else np.nan,
                              y3=np.mean(1000 / sl) if len(sl) else np.nan))
    riders = pd.concat([riders.reset_index(drop=True), riders.apply(feats, axis=1).reset_index(drop=True)], axis=1)
    riders["session_number"] = riders.groupby("username").cumcount() + 1
    riders["hour_of_day"] = riders.naive.dt.hour + riders.naive.dt.minute / 60
    riders["date"] = riders.naive.dt.date; riders["ts"] = to_unix(riders.naive)
    riders["week"] = riders.naive.dt.strftime("%G-%V")
    rn = [resolve(u, d) for u, d in zip(riders.username, riders.date)]
    riders["node"] = [x[0] for x in rn]; riders["city"] = [x[1] for x in rn]
    for c in ["pm", "temp", "rh", "shift", "pm1h", "pm4h", "pm_post", "pm_lag1", "pm_lag2", "pm_lag3", "pm_lead2"]: riders[c] = np.nan
    for i, r in riders.iterrows():
        nd = node(r.node) if r.node else None
        if nd is None: continue
        d0 = day_start_unix(r.date); d1 = d0 + 86400
        m, n, first = nd.win(d0, r.ts)
        if n >= 60:
            riders.at[i, "pm"] = m; riders.at[i, "temp"] = nd.win(d0, r.ts, "temp")[0]; riders.at[i, "rh"] = nd.win(d0, r.ts, "rh")[0]
            riders.at[i, "shift"] = (r.ts - first) / 60
        if nd.win(r.ts - 3600, r.ts)[1] >= 60: riders.at[i, "pm1h"] = nd.win(r.ts - 3600, r.ts)[0]
        if nd.win(r.ts - 4 * 3600, r.ts)[1] >= 60: riders.at[i, "pm4h"] = nd.win(r.ts - 4 * 3600, r.ts)[0]
        mp, npost, _ = nd.win(r.ts, d1)
        if npost >= 60: riders.at[i, "pm_post"] = mp
        ml, nl, _ = nd.win(d0 + 2 * 86400, d0 + 3 * 86400)
        if nl >= 600: riders.at[i, "pm_lead2"] = ml
        for L in (1, 2, 3):
            ml, nl, _ = nd.win(d0 - L * 86400, d0 - (L - 1) * 86400)
            if nl >= 600: riders.at[i, f"pm_lag{L}"] = ml
    ctrl = ["temp", "rh", "shift", "session_number", "hour_of_day"]
    P(f"# Results v3 (Asia/Bangkok explicit). Riders: {riders.username.nunique()}, sessions: {len(riders)}, with same-day PM: {riders.pm.notna().sum()} ({100*riders.pm.notna().mean():.1f}%)")
    P(f"Cities: {riders.groupby('city').username.nunique().to_dict()}")

    def run(df, y, x, controls, label, groups="date", scale=10, min_n=15, bay=True, harm=-1):
        res = []
        for u, g in df.groupby("username"):
            f = rider_fit(g, y, x, controls, groups, min_n)
            if f: res.append(f)
        if len(res) < 5: P(f"| {label} | too few riders ({len(res)}) |"); return None
        b = np.array([r["beta"] for r in res]); s = np.array([r["se"] for r in res])
        pl = pool(b, s); out_ = dict(label=label, **pl, n=sum(r["n"] for r in res))
        if bay:
            bz = np.array([r["beta"] / r["sd_y"] for r in res]) * scale; sz = np.array([r["se"] / r["sd_y"] for r in res]) * scale
            out_["bayes"] = bayes(bz, sz); out_["bayes"]["p_harm"] = out_["bayes"]["p_harm"] if harm < 0 else 1 - out_["bayes"]["p_harm"]
        return out_
    def row(o, sc=10, unit=""):
        if not o: return
        by = f" | {o['bayes']['p_harm']:.2f}" if "bayes" in o else " | "
        P(f"| {o['label']} | {o['k']} / {o['n']} | {o['mu']*sc:+.4f} | [{o['lo_hk']*sc:+.4f}, {o['hi_hk']*sc:+.4f}] | {o['p_hk']:.3f} | {o['p_dl']:.3f} | {o['i2']:.0f} | {o['share_neg']*100:.0f}%{by} |")
    HDR = "| Model | riders / n | β per +10 µg/m³ (HKSJ pooled) | 95% CI (HKSJ) | p HKSJ | p DL | I² % | riders β<0 | P(harm) Bayes |\n|---|---|---|---|---|---|---|---|---|"

    P("\n## Validation: does this build reproduce the corrected primary (FINDINGS.md: β≈−0.00012 /µg, p≈0.47)?")
    P(HDR)
    o = run(riders, "y1", "pm", ctrl, "Y1 mean speed, same-day (clean PM, 0–500)"); row(o); prim = o

    P("\n## Table 1: confirmatory PVT family (Holm across Y1–Y3 on HKSJ p)")
    P(HDR)
    fam = {}
    for y, lab in [("y1", "Y1 mean speed (Hz)"), ("y2", "Y2 lapse % (≥2× own median RT)"), ("y3", "Y3 slowest-10% speed (Hz)")]:
        fam[y] = run(riders, y, "pm", ctrl, lab, harm=(1 if y == "y2" else -1)); row(fam[y])
    ps = [fam[y]["p_hk"] for y in fam if fam[y]]; adj = holm(np.array(ps))
    P("Holm-adjusted HKSJ p: " + ", ".join(f"{k}: {v:.3f}" for k, v in zip([y for y in fam if fam[y]], adj)))

    P("\n## Table 2: sensitivity and falsification for Y1 (and lapse variants)")
    P(HDR)
    row(run(riders, "y1", "pm1h", ctrl, "Y1, preceding 1 h"))
    row(run(riders, "y1", "pm4h", ctrl, "Y1, preceding 4 h"))
    for L in (1, 2, 3): row(run(riders, "y1", f"pm_lag{L}", ctrl, f"Y1, lag {L} day mean (+ same-day controls)"))
    row(run(riders, "y1", "pm_post", ctrl, "PLACEBO Y1 vs PM after the test (rest of day)"))
    row(run(riders, "y1", "pm_lead2", ctrl, "PLACEBO Y1 vs PM two days LATER (lead 2)"))
    row(run(riders, "y1", "pm", ctrl + ["pm_post"], "Y1 same-day PM, adjusting for PM after the test"))
    row(run(riders, "y2_500", "pm", ctrl, "Y2 lapse % (abs ≥500 ms)", harm=1)); row(run(riders, "y2_1000", "pm", ctrl, "Y2 lapse % (abs ≥1000 ms)", harm=1))
    for c in sorted(riders.city.dropna().unique()):
        row(run(riders[riders.city == c], "y1", "pm", ctrl, f"Y1, city = {c}"))
    # within-week pooled FE
    def within_week(y):
        d = riders.dropna(subset=[y, "pm"] + ctrl).copy(); d["g"] = d.username + "|" + d.week
        cols = [y, "pm"] + ctrl; dm = d[cols] - d.groupby("g")[cols].transform("mean")
        keep = d.groupby("g").g.transform("size") > 1
        X = dm.loc[keep, ["pm"] + ctrl].to_numpy(); yy = dm.loc[keep, y].to_numpy()
        b, se, _ = ols(yy, X, d.loc[keep, "username"].to_numpy(), min_g=5)
        G = d.loc[keep, "username"].nunique(); t = b[0] / se[0]
        return b[0], se[0], p_t(t, G - 1), t_crit(G - 1), G, keep.sum()
    for y, lab in [("y1", "Y1"), ("y2", "Y2"), ("y3", "Y3")]:
        b, se, p, tc, G, n = within_week(y)
        P(f"| WITHIN-WEEK (rider×week FE, pooled) {lab} | {G} / {n} | {b*10:+.4f} | [{(b-tc*se)*10:+.4f}, {(b+tc*se)*10:+.4f}] | {p:.3f} | – | – | – | – |")

    # ---- Y4 same-day pairs ----
    P("\n## Table 3 (exploratory): same-day change between consecutive tests (Y4)")
    P(HDR)
    prs = []
    for (u, dte), g in riders.groupby(["username", "date"]):
        g = g.sort_values("ts"); nd = node(g.node.iloc[0]) if g.node.iloc[0] else None
        if nd is None or len(g) < 2: continue
        for i in range(len(g) - 1):
            r1, r2 = g.iloc[i], g.iloc[i + 1]
            m, n, _ = nd.win(r1.ts, r2.ts)
            if n < 60: continue
            prs.append(dict(username=u, date=dte, dy=r2.y1 - r1.y1, dy3=r2.y3 - r1.y3, dpm=m, el=(r2.ts - r1.ts) / 3600, h2=r2.hour_of_day,
                            t=nd.win(r1.ts, r2.ts, "temp")[0], rh=nd.win(r1.ts, r2.ts, "rh")[0], first_last=0))
    pr = pd.DataFrame(prs); P(f"pairs with exposure between tests: {len(pr)} from {pr.username.nunique()} riders")
    row(run(pr, "dy", "dpm", ["el", "h2", "t", "rh"], "Y4 Δ mean speed vs PM between tests"))
    row(run(pr, "dy3", "dpm", ["el", "h2", "t", "rh"], "Y4 Δ slowest-10% speed vs PM between tests"))
    fl = pr.sort_values(["username", "date"]).groupby(["username", "date"]).agg(dy=("dy", "sum"), dpm=("dpm", "mean"), el=("el", "sum"), h2=("h2", "last"), t=("t", "mean"), rh=("rh", "mean")).reset_index()
    row(run(fl, "dy", "dpm", ["el", "h2", "t", "rh"], "Y4 first-to-last of day (sum of Δ)", groups=None))

    # ---- Peak flow ----
    P("\n## Table 4: peak flow, % of personal best (Y5)")
    P(HDR)
    pef = pd.read_csv(os.path.join(a.data_dir, "peakflow.csv")); pef["username"] = pef.username.astype(str).str.strip()
    pef = pef[~pef.username.isin(STAFF)].copy(); pef["naive"] = pd.to_datetime(pef.timestamp)
    pef = pef.drop_duplicates(["username", "naive"]); n0 = len(pef); pef = pef[pef.peak_flow >= 50].copy()
    P(f"peak-flow readings kept {len(pef)} of {n0} (dropped <50 L/min)")
    pb = pef.groupby("username").peak_flow.quantile(0.95); pef["pct"] = 100 * pef.peak_flow / pef.username.map(pb)
    pef["workday"] = (pef.naive - pd.Timedelta(hours=4)).dt.date; h = pef.naive.dt.hour
    pef["slot"] = np.where((h >= 4) & (h < 13), "morning", np.where((h >= 15) | (h < 4), "evening", "other"))
    pef["ts"] = to_unix(pef.naive)
    mo = pef[pef.slot == "morning"].sort_values("naive").groupby(["username", "workday"]).first().reset_index()
    ev = pef[pef.slot == "evening"].sort_values("naive").groupby(["username", "workday"]).last().reset_index()
    P(f"workdays with morning reading {len(mo)}, evening reading {len(ev)}")
    studyday = pef.groupby("username").naive.min().dt.normalize()
    def attach(df, slot):
        df = df.copy()
        for c in ["pm", "temp", "rh", "pm_lag", "temp_lag", "rh_lag", "pre", "shift", "pm_lead2"]: df[c] = np.nan
        df["city"] = None
        for i, r in df.iterrows():
            nr = resolve(r.username, r.workday); df.at[i, "city"] = nr[1]
            nd = node(nr[0]) if nr[0] else None
            if nd is None: continue
            d0 = day_start_unix(r.workday)
            m, n, first = nd.win(d0, r.ts)
            if slot == "evening" and n >= 60:
                df.at[i, "pm"] = m; df.at[i, "temp"] = nd.win(d0, r.ts, "temp")[0]; df.at[i, "rh"] = nd.win(d0, r.ts, "rh")[0]; df.at[i, "shift"] = (r.ts - first) / 60
                ml2, nl2, _ = nd.win(d0 + 2 * 86400, d0 + 3 * 86400)
                if nl2 >= 600: df.at[i, "pm_lead2"] = ml2
            if slot == "morning":
                df.at[i, "pre"] = 1.0 if (n == 0 or (r.ts - first) < 15 * 60) else 0.0
                ml, nl, _ = nd.win(d0 - 86400, d0)
                if nl >= 600:
                    df.at[i, "pm_lag"] = ml; df.at[i, "temp_lag"] = nd.win(d0 - 86400, d0, "temp")[0]; df.at[i, "rh_lag"] = nd.win(d0 - 86400, d0, "rh")[0]
        df["studyday"] = (df.naive - df.username.map(studyday)).dt.days
        df["date"] = df.workday
        return df
    ev = attach(ev, "evening"); mo = attach(mo, "morning")
    P(f"morning readings with NO sensor data >15 min earlier the same day (spec rule for Y5b): {int((mo.pre==1).sum())} of {len(mo)}; with previous-calendar-day exposure: {int(mo.pm_lag.notna().sum())}  => Y5b NOT FEASIBLE as specified (riders test every ~2 days; sensor is on at home before the morning blow)")
    row(run(ev, "pct", "pm", ["temp", "rh", "studyday"], "Y5a evening %PB vs same-day cumulative PM", groups=None, min_n=12))
    row(run(ev, "pct", "pm", ["temp", "rh", "studyday", "shift"], "Y5a + shift minutes", groups=None, min_n=12))
    row(run(ev, "pct", "pm_lead2", ["temp", "rh", "studyday"], "PLACEBO Y5a vs PM two days LATER", groups=None, min_n=12))
    evw = ev.dropna(subset=["pct", "pm", "temp", "rh", "studyday"]).copy(); evw["g"] = evw.username + "|" + pd.to_datetime(evw.workday).dt.strftime("%G-%V")
    cols = ["pct", "pm", "temp", "rh", "studyday"]; dm = evw[cols] - evw.groupby("g")[cols].transform("mean"); kp = evw.groupby("g").g.transform("size") > 1
    bw, sw, _ = ols(dm.loc[kp, "pct"].to_numpy(), dm.loc[kp, ["pm", "temp", "rh", "studyday"]].to_numpy(), evw.loc[kp, "username"].to_numpy(), min_g=5)
    Gw = evw.loc[kp, "username"].nunique(); tcw = t_crit(Gw - 1)
    P(f"| WITHIN-WEEK (rider×week FE, pooled) Y5a | {Gw} / {int(kp.sum())} | {bw[0]*10:+.4f} | [{(bw[0]-tcw*sw[0])*10:+.4f}, {(bw[0]+tcw*sw[0])*10:+.4f}] | {p_t(bw[0]/sw[0], Gw-1):.3f} | – | – | – | – |")
    row(run(mo[mo.pre == 1], "pct", "pm_lag", ["temp_lag", "rh_lag", "studyday"], "Y5b morning %PB vs previous-day mean PM", groups=None, min_n=12))
    ch = ev.merge(mo[["username", "workday", "pct", "pre"]], on=["username", "workday"], suffixes=("_ev", "_mo"))
    ch["dpct"] = ch.pct_ev - ch.pct_mo; ch["date"] = ch.workday
    P(f"workdays with both morning and evening readings: {len(ch)}")
    row(run(ch, "dpct", "pm", ["temp", "rh", "studyday"], "Y5c evening − morning %PB vs same-day PM", groups=None, min_n=12))
    for c in sorted(ev.city.dropna().unique()):
        row(run(ev[ev.city == c], "pct", "pm", ["temp", "rh", "studyday"], f"Y5a evening, city = {c}", groups=None, min_n=12))
    P(f"descriptive: mean %PB morning {mo.pct.mean():.1f}, evening {ev.pct.mean():.1f}; share of evening readings <80% PB {100*(ev.pct<80).mean():.1f}%, morning {100*(mo.pct<80).mean():.1f}%")
    if ev.pm.notna().sum() > 30:
        q = ev.dropna(subset=["pm"]).copy(); q["tert"] = pd.qcut(q.pm, 3, labels=["low", "mid", "high"])
        P("share of evening readings <80% personal best by PM tertile: " + ", ".join(f"{k}: {100*(g.pct<80).mean():.1f}% (n={len(g)})" for k, g in q.groupby("tert")))

    # ---- Office-worker contrast ----
    P("\n## Table 5: personal PM2.5, riders vs office workers (descriptive; matched city, date, clock hours; >=10 min per hour)")
    OFFICE_NODES = {"adisorn": ("Grab-6010", "Bangkok"), "Preechai": ("Grab-6009", None), "Nussara": ("Grab-6033", "Bangkok")}
    def hourly_df(path, default_city):
        d = pd.read_csv(path, usecols=["ts", "pm", "lat"], dtype={"ts": "int64"})
        d = d[(d.pm >= 0) & (d.pm <= 500)]; d["hr"] = (d.ts + OFFSET) // 3600
        h = d.groupby("hr").agg(pm=("pm", "mean"), n=("pm", "size"), lat=("lat", "median")); h = h[h.n >= 600]
        h["city"] = np.where(h.lat.isna(), default_city, np.where(h.lat < 16, "Bangkok", "Chiang mai"))
        h["t"] = pd.to_datetime(h.index * 3600, unit="s"); h["date"] = h.t.dt.date; return h.dropna(subset=["city"])
    # rider city-hours, restricted to each rider's roster deployment period
    rows_ = []
    for nname, grp in roster.groupby("node"):
        pth = os.path.join(a.nodes_dir, f"{nname}.csv")
        if not os.path.exists(pth): continue
        h = hourly_df(pth, grp.city.iloc[0]); h = h[h.apply(lambda r: any(rr.sd <= r.date <= rr.ed for rr in grp.itertuples()), axis=1)]
        rows_.append(h.assign(u=nname))
    rid_h = pd.concat(rows_); rid_h = rid_h[rid_h.city == rid_h.city]
    for ou, (nname, dc) in OFFICE_NODES.items():
        pth = os.path.join(a.nodes_dir, f"{nname}.csv")
        if not os.path.exists(pth): P(f"{ou}: node file {nname} missing"); continue
        oh = hourly_df(pth, dc)
        if ou == "Nussara":
            pvd = set(pvt[pvt.username.str.lower() == "nussara"].naive.dt.date)
            pf_ = pd.read_csv(os.path.join(a.data_dir, "peakflow.csv")); pfd = set(pd.to_datetime(pf_[pf_.username.astype(str).str.strip().str.lower() == "nussara"].timestamp).dt.date)
            keep = set(oh.date) & pvd & pfd; oh = oh[oh.date.isin(keep)]
            P(f"Nussara: dates with PVT + peak flow + AQ all present = {len(keep)} (of AQ dates {len(set(hourly_df(pth, dc).date))}, PVT dates {len(pvd)}, peak-flow dates {len(pfd)})")
            if not keep: continue
        mm = oh.reset_index().merge(rid_h.reset_index(), on=["hr", "city"], suffixes=("_o", "_r"))
        if mm.empty: P(f"{ou}: no overlapping city-hours with riders"); continue
        per = mm.groupby(["date_o", "hr"]).agg(o=("pm_o", "first"), r=("pm_r", "mean"), nr=("u", "nunique"), city=("city", "first")).reset_index()
        daily = per.groupby("date_o").agg(o=("o", "mean"), r=("r", "mean"), hours=("hr", "count")).reset_index()
        d = daily.r - daily.o
        if len(d) > 2:
            se = d.std(ddof=1) / math.sqrt(len(d)); tcv = t_crit(len(d) - 1); ci = f"[{d.mean()-tcv*se:+.1f}, {d.mean()+tcv*se:+.1f}]"
        else: ci = "n/a"
        P(f"{ou} ({', '.join(sorted(per.city.unique()))}): {len(daily)} matched dates, {len(per)} matched city-hours; office mean {daily.o.mean():.1f}, riders mean {daily.r.mean():.1f} µg/m³; rider − office = {d.mean():+.1f} {ci}; ratio of means {daily.r.mean()/daily.o.mean():.2f}; share of matched hours where riders higher {100*(per.r > per.o).mean():.0f}%")
        daily.to_csv(os.path.join(a.out, f"office_contrast_{ou.lower()}.csv"), index=False)

    open(os.path.join(a.out, "results_v3.md"), "w").write("\n".join(out) + "\n")
    riders.drop(columns=["rts", "response_times"]).to_csv(os.path.join(a.out, "pvt_linked_bangkok.csv"), index=False)

if __name__ == "__main__":
    main()
