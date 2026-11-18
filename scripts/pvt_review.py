"""SPEC_ADDENDUM §P: review comments 2 and 3 (written 2026-11-17 before computing). Run once. Exploratory except P2 row e.

Usage: python3 -I scripts/pvt_review.py [--out pvt_paper/run_review]
Inputs (grant G3 + §C files): foundation/linked_v4_pvt.csv; foundation/_work/aq_min/<node>.csv (per-minute sums, used only to
rebuild the OLD misaligned window for P2 rows a0/a); data/review_2026_09/metar_VTBD.csv, metar_VTCC.csv (from §C).
Clock rule: no .timestamp() on naive times. The old window is rebuilt as REVIEW_2026-09-28 §1: [Los Angeles midnight of the
test's calendar date, test end read as UTC (= true end + 7 h)], capped at LA end of day. pandas tz_localize is explicit.
Helpers copied unchanged from scripts/pvt_heat.py (itself copied from pvt_analysis.py).
"""
import argparse, json, math, os
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser()
ap.add_argument("--out", default=os.path.join(ROOT, "pvt_paper", "run_review"))
A = ap.parse_args(); os.makedirs(A.out, exist_ok=True)

# ---------- statistics (copied from pvt_analysis.py) ----------
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
def p_t(t, df): return betainc(df / 2, 0.5, df / (df + t * t))
def p_z(z): return math.erfc(abs(z) / math.sqrt(2))
def t_crit(df, q=0.975):
    lo, hi = 0.0, 80.0
    for _ in range(90):
        mid = (lo + hi) / 2
        if 1 - 0.5 * p_t(mid, df) < q: lo = mid
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
    return b, np.sqrt(np.maximum(np.diag(c * XtXi @ meat @ XtXi), 0))
def pool(betas, ses, q=0.975):
    b = np.asarray(betas, float); s = np.asarray(ses, float); k = len(b)
    if k < 3: return None
    w = 1 / s ** 2; mu_f = (w * b).sum() / w.sum(); Q = (w * (b - mu_f) ** 2).sum()
    tau2 = max(0.0, (Q - (k - 1)) / (w.sum() - (w ** 2).sum() / w.sum()))
    ws = 1 / (s ** 2 + tau2); mu = (ws * b).sum() / ws.sum(); se_dl = math.sqrt(1 / ws.sum())
    var_hk = (ws * (b - mu) ** 2).sum() / ((k - 1) * ws.sum()); se_hk = math.sqrt(max(var_hk, 1 / ws.sum()))
    tc = t_crit(k - 1, q)
    return dict(k=k, mu=mu, lo_hk=mu - tc * se_hk, hi_hk=mu + tc * se_hk, p_hk=p_t(mu / se_hk, k - 1),
                se_hk=se_hk, se_dl=se_dl, p_dl=p_z(mu / se_dl), tau2=tau2,
                i2=max(0.0, (Q - (k - 1)) / Q) * 100 if Q > 0 else 0.0, n_neg=int((b < 0).sum()))
def tobool(s):
    return s.map(lambda v: True if v in (True, "True", 1, "1") else (False if v in (False, "False", 0, "0") else np.nan))
def zs(x):
    sd = x.std(ddof=1); return (x - x.mean()) / sd if sd > 0 else x * np.nan


OUT = []; NM = [0]; RES = {}
def P(s=""): OUT.append(s)
def f2(mu, lo, hi, p, d=2): return f"{mu:+.{d}f} [{lo:+.{d}f}, {hi:+.{d}f}], p {p:.3f}"

L = pd.read_csv(os.path.join(ROOT, "foundation", "linked_v4_pvt.csv"))
L["ts"] = pd.to_datetime(L.timestamp_end_local)
L = L.sort_values(["pid", "ts"]).reset_index(drop=True)
inenr = ~tobool(L.P11_outside_enrolment).fillna(False).astype(bool)
D = L[(L.role == "rider") & inenr].copy()
D["Y1"] = 100 * (D.y1_speed_rt100 / D.groupby("pid").y1_speed_rt100.transform("mean") - 1)
D["Ys"] = 100 * (D.y1_speed_rt100 / D.y1_speed_rt100.mean() - 1)          # % of cohort mean speed (pooled rows)
D["days30"] = D.days_since_enrolment / 30.0
D["cnt10"] = D.session_number / 10.0
D["PMc"] = D.pm_sameday.where(tobool(D.cov_ok_sameday).fillna(False).astype(bool))
D["PMa"] = D.pm_after.where(tobool(D.cov_ok_after).fillna(False).astype(bool))

# ---------- old (misaligned) window rebuilt from per-minute sums ----------
def old_window(df):
    res = pd.DataFrame(index=df.index, columns=["PMo", "To", "Ho", "shift_o"], dtype=float)
    end_bug = df.test_end_unix.to_numpy(float) + 7 * 3600            # naive end read as UTC
    cal = df.ts.dt.normalize()
    lo = (cal.dt.tz_localize("America/Los_Angeles").dt.tz_convert("UTC").dt.tz_localize(None) - pd.Timestamp("1970-01-01")).dt.total_seconds().to_numpy()
    hi = np.minimum(lo + 86400 - 1, end_bug)
    for node, idx in df.groupby("node").groups.items():
        f = os.path.join(ROOT, "foundation", "_work", "aq_min", f"{int(node)}.csv")
        if not os.path.exists(f): continue
        m = pd.read_csv(f).sort_values("minute_unix"); t = m.minute_unix.to_numpy(float)
        cs = {c: np.concatenate([[0], np.cumsum(m[c].to_numpy(float))]) for c in ["n_pm", "pm_sum", "n_t", "t_sum", "n_rh", "rh_sum"]}
        pos = df.index.get_indexer(idx)
        a = np.searchsorted(t, lo[pos], "left"); b = np.searchsorted(t, hi[pos], "right")
        n = cs["n_pm"][b] - cs["n_pm"][a]
        ok = (b > a) & (n > 0)
        S = lambda c: cs[c][b] - cs[c][a]
        res.loc[idx, "PMo"] = np.where(ok, S("pm_sum") / np.where(n > 0, n, 1), np.nan)
        nt = S("n_t"); nh = S("n_rh")
        res.loc[idx, "To"] = np.where(ok & (nt > 0), S("t_sum") / np.where(nt > 0, nt, 1), np.nan)
        res.loc[idx, "Ho"] = np.where(ok & (nh > 0), S("rh_sum") / np.where(nh > 0, nh, 1), np.nan)
        first = t[np.minimum(a, len(t) - 1)]
        res.loc[idx, "shift_o"] = np.where(ok, (hi[pos] - first) / 60, np.nan)
    return res
OW = old_window(L[L.role == "rider"])
L = L.join(OW); D = D.join(OW)
P("## Diagnostics")
P(f"- Old-window coverage, all rider tests: {L[L.role=='rider'].PMo.notna().sum()}/{(L.role=='rider').sum()} (old build: 1,755 sessions, 91.3%)")
P(f"- Riders in-enrolment sample: {len(D)} tests; corrected same-day PM {D.PMc.notna().sum()}, old-window PM {D.PMo.notna().sum()}, PM after {D.PMa.notna().sum()}")
w = D.dropna(subset=["PMo", "PMc"])
dm = lambda x: x - x.groupby(w.pid).transform("mean")
P(f"- within-rider r(old-window PM, corrected PM) {np.corrcoef(dm(w.PMo), dm(w.PMc))[0,1]:+.2f}")
wa = w.dropna(subset=["PMa"]); P(f"- within-rider r(old-window PM, PM after the test) {np.corrcoef(wa.PMo - wa.groupby('pid').PMo.transform('mean'), wa.PMa - wa.groupby('pid').PMa.transform('mean'))[0,1]:+.2f}")

# ---------- a0: PLAN §4 as originally run (old clock, Hz, RT>0, CR1 by date, DL + Wald) ----------
R = L[L.role == "rider"].copy()
R["date"] = (R.ts - pd.Timedelta(hours=4)).dt.date.astype(str)
rows = []
for pid, d in R.dropna(subset=["PMo", "To", "Ho", "shift_o", "clock_hour", "y1_speed_rtgt0"]).groupby("pid"):
    if len(d) < 15: continue
    X = np.column_stack([np.ones(len(d)), d.PMo, d.To, d.Ho, d.shift_o, d.session_number, d.clock_hour]).astype(float)
    b, se = ols(d.y1_speed_rtgt0.to_numpy(float), X, d.date.to_numpy(), min_g=2)
    rows.append((b[1], se[1], len(d)))
pa0 = pool([r[0] for r in rows], [r[1] for r in rows]); NM[0] += 1
P(f"- **a0 reproduction** (PLAN §4 old clock, Hz per µg/m³, DL + Wald z): {pa0['mu']:+.5f} [{pa0['mu']-1.96*pa0['se_dl']:+.5f}, {pa0['mu']+1.96*pa0['se_dl']:+.5f}], "
  f"p {pa0['p_dl']:.3f}; k {pa0['k']}, n {sum(r[2] for r in rows)}. Target −0.00064 [−0.00125, −0.00003], p 0.040, n 1,755. HKSJ p {pa0['p_hk']:.3f}")
RES["a0"] = dict(mu=pa0["mu"], p_dl=pa0["p_dl"], p_hk=pa0["p_hk"], k=pa0["k"])

# ---------- pooled random-intercept GLS (Swamy–Arora), CR1 by rider ----------
def re_gls(df, y, xs, focus):
    d = df.dropna(subset=[y] + xs).copy(); g = d.pid.to_numpy()
    Y = d[y].to_numpy(float); X = d[xs].to_numpy(float)
    Ti = d.groupby("pid")[y].transform("size").to_numpy(float)
    gm = lambda a: pd.DataFrame(a, index=d.index).groupby(g).transform("mean").to_numpy()
    Yw = Y - gm(Y[:, None])[:, 0]; Xw = X - gm(X)
    keep = Xw.std(0) > 1e-9                                     # time-invariant columns drop from the within fit
    bw = np.linalg.lstsq(Xw[:, keep], Yw, rcond=None)[0]; G = len(np.unique(g)); N = len(Y)
    s2e = ((Yw - Xw[:, keep] @ bw) ** 2).sum() / (N - G - keep.sum())
    Mb = d.assign(_y=Y).groupby("pid")[["_y"] + xs].mean()
    Xb = np.column_stack([np.ones(G), Mb[xs].to_numpy(float)]); bb = np.linalg.lstsq(Xb, Mb._y.to_numpy(), rcond=None)[0]
    s2b = ((Mb._y.to_numpy() - Xb @ bb) ** 2).sum() / max(G - Xb.shape[1], 1)
    Tg = d.groupby("pid").size().to_numpy(float)
    s2u = max(0.0, s2b - s2e * np.mean(1 / Tg))
    th = 1 - np.sqrt(s2e / (s2e + Ti * s2u))
    Ys = Y - th * gm(Y[:, None])[:, 0]; Xs = np.column_stack([1 - th, X - th[:, None] * gm(X)])
    b, se = ols(Ys, Xs, g, min_g=2)
    out = {}
    for f in focus:
        j = 1 + xs.index(f); t = b[j] / se[j]; tc = t_crit(G - 1)
        out[f] = dict(mu=b[j], lo=b[j] - tc * se[j], hi=b[j] + tc * se[j], p=p_t(t, G - 1), n=N, G=G)
    NM[0] += 1
    return out, dict(s2u=s2u, s2e=s2e, theta_med=float(np.median(th)))
def wsd(df, c):
    d = df.dropna(subset=[c]); return float(np.sqrt((d[c] - d.groupby("pid")[c].transform("mean")).pow(2).sum() / (len(d) - d.pid.nunique())))
def scaled(df, c, name):
    s = wsd(df, c); df[name] = df[c] / s; return s

LAD = []
def lad(lab, desc, r, note=""):
    LAD.append((lab, desc, r)); RES["P2_" + lab] = r
    P(f"- **{lab}** {desc}: {f2(r['mu'], r['lo'], r['hi'], r['p'])}; n {r['n']}, riders {r['G']}{note}")
P("\n## P2 Comment 2: model ladder (% of cohort mean speed per within-rider SD of PM; pooled random intercept, CR1 by rider, t21)")
Z = D.copy()
so = scaled(Z, "PMo", "PMo_s"); sc = scaled(Z, "PMc", "PMc_s"); sa = scaled(Z, "PMa", "PMa_s")
COVo = ["To", "Ho", "clock_hour", "session_number"]; COVc = ["temp_sameday", "rh_sameday", "clock_hour", "session_number"]
r, vc = re_gls(Z, "Ys", ["PMo_s"] + COVo, ["PMo_s"]); lad("a", "old clock, PLAN §5 pooled model", r["PMo_s"], f"; θ median {vc['theta_med']:.2f}")
r, vc = re_gls(Z, "Ys", ["PMc_s"] + COVc, ["PMc_s"]); lad("b", "corrected clock, same model", r["PMc_s"])
Zb = Z[Z.PMo.notna()]
r, _ = re_gls(Zb, "Ys", ["PMc_s"] + COVc, ["PMc_s"]); lad("b′", "(b) on the old-clock sessions only", r["PMc_s"])
Z["PMc_b"] = Z.groupby("pid").PMc_s.transform("mean"); Z["PMc_w"] = Z.PMc_s - Z.PMc_b
r, _ = re_gls(Z, "Ys", ["PMc_w", "PMc_b"] + COVc, ["PMc_w", "PMc_b"])
lad("c-within", "(b) + Mundlak: within-rider PM", r["PMc_w"]); lad("c-between", "(b) + Mundlak: rider-mean PM (per within-SD unit)", r["PMc_b"])
Z["PMo_b"] = Z.groupby("pid").PMo_s.transform("mean"); Z["PMo_w"] = Z.PMo_s - Z.PMo_b
r, _ = re_gls(Z, "Ys", ["PMo_w", "PMo_b"] + COVo, ["PMo_w", "PMo_b"])
lad("a-Mundlak-within", "(a) old clock + Mundlak: within-rider PM", r["PMo_w"]); lad("a-Mundlak-between", "(a) old clock + Mundlak: rider-mean PM", r["PMo_b"])
r, _ = re_gls(Z, "Ys", ["PMc_s"] + COVc + ["days30"], ["PMc_s", "days30"]); lad("d", "(b) + days since enrolment / 30", r["PMc_s"])
P(f"  - days/30 term in (d): {f2(r['days30']['mu'], r['days30']['lo'], r['days30']['hi'], r['days30']['p'])}")
r, _ = re_gls(Z, "Ys", ["PMo_s"] + COVo + ["days30"], ["PMo_s"]); lad("a+trend", "(a) old clock + days/30", r["PMo_s"])
e = json.load(open(os.path.join(ROOT, "pvt_paper", "run_raw", "pooled.json")))["pooled"]["F1_Y1"]["pm"]
P(f"- **e** two-stage DL/HKSJ headline F1-Y1 (quoted from run_raw, % of own mean): {f2(e['mu'], e['lo_hk'], e['hi_hk'], e['p_hk'])}; k {e['k']}, n {e['n']}")
r, _ = re_gls(Z, "Ys", ["PMa_s"] + COVc, ["PMa_s"]); lad("f", "placebo: PM after the test, (b) model", r["PMa_s"])

# ---------- P3 engine: per-person OLS -> DL + HKSJ ----------
PER = {}
def fit(df, y, spec, focus, label):
    cols = [y] + [c for _, c in spec]; rows = []
    for pid, d in df.dropna(subset=cols).groupby("pid"):
        if len(d) < 15: continue
        X = pd.DataFrame({lab: d[c].astype(float) for lab, c in spec}, index=d.index)
        if any(X[f].std() < 1e-9 for f in focus): continue
        X = X[[c for c in X.columns if X[c].std() > 1e-9]]
        Xm = np.column_stack([np.ones(len(d)), X.to_numpy(float)])
        if np.linalg.matrix_rank(Xm) < Xm.shape[1]: continue
        bb, se = ols(d[y].to_numpy(float), Xm, d.workday.to_numpy())
        r = dict(pid=pid, n=len(d))
        for f in focus: j = 1 + list(X.columns).index(f); r["b_" + f] = bb[j]; r["se_" + f] = se[j]
        rows.append(r)
    pr = pd.DataFrame(rows); PER[label] = pr; out = {}
    for f in focus:
        p = pool(pr["b_" + f], pr["se_" + f]); p["n"] = int(pr.n.sum()); out[f] = p
    RES[label] = out; NM[0] += 1
    return out
def fmt(p, d=2):
    return f"{p['mu']:+.{d}f} [{p['lo_hk']:+.{d}f}, {p['hi_hk']:+.{d}f}], p {p['p_hk']:.3f}; k {p['k']}, n {p['n']}, I² {p['i2']:.0f}%, β<0 in {p['n_neg']}/{p['k']}"

P("\n## P3a Count vs calendar (Y1 % of own mean speed; + clock hour)")
w = D.dropna(subset=["days30", "cnt10"])
r_dc = np.corrcoef(w.days30 - w.groupby("pid").days30.transform("mean"), w.cnt10 - w.groupby("pid").cnt10.transform("mean"))[0, 1]
P(f"- within-rider r(days, test count) {r_dc:+.2f} (VIF {1/(1-r_dc**2):.1f}); per-rider r median {D.groupby('pid').apply(lambda g: g.days30.corr(g.cnt10)).median():+.2f}")
o = fit(D, "Y1", [("days", "days30"), ("clock", "clock_hour")], ["days"], "P3a_days"); P(f"- days alone, % per 30 days: {fmt(o['days'])}")
o = fit(D, "Y1", [("cnt", "cnt10"), ("clock", "clock_hour")], ["cnt"], "P3a_cnt"); P(f"- count alone, % per 10 tests: {fmt(o['cnt'])}")
o = fit(D, "Y1", [("days", "days30"), ("cnt", "cnt10"), ("clock", "clock_hour")], ["days", "cnt"], "P3a_both")
P(f"- **both: days % per 30 d {fmt(o['days'])}**"); P(f"- **both: count % per 10 tests {fmt(o['cnt'])}**")
P(f"- tests per 30 days, median rider: {(D.groupby('pid').session_number.max() / D.groupby('pid').days30.max()).median():.1f}")

P("\n## P3b Signatures per 30 days (days term; + clock hour). Units: as stated; mean level in brackets")
D["valid_ratio"] = 100 * D.valid_taps / D.total_taps
D["medrt_pct"] = 100 * (D.med_rt / D.groupby("pid").med_rt.transform("mean") - 1)
D["Y3"] = 100 * (D.y3_slowest10_speed / D.groupby("pid").y3_slowest10_speed.transform("mean") - 1)
for y, lab in [("Y1", "mean speed, % of own mean"), ("medrt_pct", "median RT, % of own mean (+ = slower)"), ("Y3", "slowest-10% speed, % of own mean"),
               ("y2_lapse_pct_500", "lapses RT ≥ 500 ms, pp"), ("false_starts", "false starts per test"), ("n_anticip", "anticipations RT < 100 ms per test"),
               ("valid_ratio", "valid/total taps, pp"), ("percent_success", "percent_success, pp")]:
    o = fit(D, y, [("days", "days30"), ("clock", "clock_hour")], ["days"], f"P3b_{y}")
    P(f"- {lab} [{D[y].mean():.2f}]: {fmt(o['days'], 3)}")
P("- Fastest-10% RT and within-test decrement: NOT RUN (trial-level data/pvt.csv is outside grant G3).")
M = D.assign(mon=D.ts.dt.to_period("M").astype(str)).groupby("mon").agg(n=("Y1", "size"), Y1=("Y1", "mean"), lapse=("y2_lapse_pct_500", "mean"),
      fs=("false_starts", "mean"), antic=("n_anticip", "mean"), succ=("percent_success", "mean"), medrt=("medrt_pct", "mean"), y3=("Y3", "mean"))
P("\n| Month | n | Y1 % | median RT % | slowest-10% % | lapse % | false starts | anticip. | success % |\n|---|---|---|---|---|---|---|---|---|")
for k_, r_ in M[M.n >= 20].iterrows():
    P(f"| {k_} | {r_.n} | {r_.Y1:+.1f} | {r_.medrt:+.1f} | {r_.y3:+.1f} | {r_.lapse:.1f} | {r_.fs:.2f} | {r_.antic:.2f} | {r_.succ:.1f} |")

P("\n## P3c Office workers as a control (days since own first test / 30; + clock hour)")
O = L[L.role == "office"].copy()
O["Y1"] = 100 * (O.y1_speed_rt100 / O.groupby("pid").y1_speed_rt100.transform("mean") - 1)
O["days30"] = (O.ts - O.groupby("pid").ts.transform("min")).dt.total_seconds() / 86400 / 30
o = fit(O, "Y1", [("days", "days30"), ("clock", "clock_hour")], ["days"], "P3c_office")
P(f"- office pooled % per 30 d: {fmt(o['days'])}")
for _, r_ in PER["P3c_office"].iterrows():
    P(f"  - {r_.pid}: {r_.b_days:+.2f} (SE {r_.se_days:.2f}), n {r_.n}, span {O[O.pid==r_.pid].days30.max()*30:.0f} d")
P(f"- office tests not modelled: {', '.join(f'{p} (n {n})' for p, n in O.groupby('pid').size().items() if p not in set(PER['P3c_office'].pid))}")
D["days30_first"] = (D.ts - D.groupby("pid").ts.transform("min")).dt.total_seconds() / 86400 / 30
o = fit(D, "Y1", [("days", "days30_first"), ("clock", "clock_hour")], ["days"], "P3c_riders_samedef")
P(f"- riders, same definition (days since own first test): {fmt(o['days'])}")

P("\n## P3d Heat per month (METAR at test start, ±90 min, Bangkok local both sides)")
D["t_start"] = D.ts - pd.Timedelta(seconds=300); D["amb"] = np.nan
for city, st in [("BKK", "VTBD"), ("CMI", "VTCC")]:
    m = pd.read_csv(os.path.join(ROOT, "data", "review_2026_09", f"metar_{st}.csv"), na_values="M").dropna(subset=["tmpc"])
    m["key"] = pd.to_datetime(m.valid); m = m.sort_values("key")[["key", "tmpc"]]
    idx = D.city == city
    q = D.loc[idx, ["t_start"]].assign(key=D.loc[idx, "t_start"], row=D.index[idx]).sort_values("key")
    mm = pd.merge_asof(q, m, on="key", direction="nearest", tolerance=pd.Timedelta(minutes=90))
    D.loc[mm["row"].to_numpy(), "amb"] = mm.tmpc.to_numpy()
P(f"- coverage {D.amb.notna().mean():.1%}")
MM = D.assign(mon=D.ts.dt.to_period("M").astype(str)).groupby(["city", "mon"]).agg(n=("Y1", "size"), Y1=("Y1", "mean"), amb=("amb", "mean"))
P("\n| City | Month | n | Y1 % of own mean | METAR °C at test |\n|---|---|---|---|---|")
for (c_, k_), r_ in MM[MM.n >= 20].iterrows(): P(f"| {c_} | {k_} | {r_.n} | {r_.Y1:+.1f} | {r_.amb:.1f} |")
s = D.dropna(subset=["amb", "days30", "clock_hour"])
o0 = fit(s, "Y1", [("days", "days30"), ("clock", "clock_hour")], ["days"], "P3d_days")
o1 = fit(s, "Y1", [("days", "days30"), ("amb", "amb"), ("clock", "clock_hour")], ["days", "amb"], "P3d_days_amb")
P(f"- days/30 without temperature: {fmt(o0['days'])}")
P(f"- days/30 with temperature (°C): {fmt(o1['days'])}; temperature per °C {fmt(o1['amb'], 3)}")
P(f"- share of the days slope removed by temperature: {1 - abs(o1['days']['mu']) / abs(o0['days']['mu']):.0%}")
for c_ in ["BKK", "CMI"]:
    o = fit(s[s.city == c_], "Y1", [("days", "days30"), ("clock", "clock_hour")], ["days"], f"P3d_{c_}")
    P(f"- {c_} days/30: {fmt(o['days'])}")

P("\n## P3e Breaks of ≥ 7 days between a rider's tests (mean Y1 of 3 tests after minus 3 before)")
br = []
for pid, d in D.groupby("pid"):
    d = d.sort_values("ts").reset_index(drop=True); gap = d.ts.diff().dt.total_seconds() / 86400
    for i in np.where(gap >= 7)[0]:
        if i >= 3 and i + 3 <= len(d):
            br.append(dict(pid=pid, gap=gap[i], diff=d.Y1.iloc[i:i+3].mean() - d.Y1.iloc[i-3:i].mean(), date=str(d.ts.iloc[i].date())))
B_ = pd.DataFrame(br)
slope = RES["P3a_days"]["days"]["mu"]
if len(B_):
    pr = B_.groupby("pid").agg(diff=("diff", "mean"), gap=("gap", "mean"), nb=("diff", "size"))
    k = len(pr); mu = pr["diff"].mean(); se = pr["diff"].std(ddof=1) / np.sqrt(k); tc = t_crit(k - 1)
    P(f"- {len(B_)} breaks in {k} riders; gap median {B_.gap.median():.0f} d (range {B_.gap.min():.0f}–{B_.gap.max():.0f}); break dates (first test after): "
      + ", ".join(f"{d_}" for d_ in sorted(B_.date.unique())[:12]))
    P(f"- **after − before, rider means: {mu:+.2f}% [{mu - tc*se:+.2f}, {mu + tc*se:+.2f}], p {p_t(mu/se, k-1):.3f}** (positive = faster after the break)")
    P(f"- calendar-time prediction for the gap (days slope × gap/30): {slope * B_.gap.mean() / 30:+.2f}%; repetition prediction ≈ 0 or recovery (> 0)")
    B_.to_csv(os.path.join(A.out, "breaks.csv"), index=False)
P(f"\n## Model count: {NM[0]} this script")
open(os.path.join(A.out, "results_review.md"), "w").write("\n".join(OUT) + "\n")
json.dump({k: v for k, v in RES.items()}, open(os.path.join(A.out, "pooled_review.json"), "w"), indent=1, default=float)
print("\n".join(OUT))
