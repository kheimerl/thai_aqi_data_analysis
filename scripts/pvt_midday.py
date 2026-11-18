"""SPEC_ADDENDUM §R: midday (2nd) test and "best taps" (fastest 10%). Written 2026-11-18 before computing. EXPLORATORY SCREEN.
Usage: python3 -I scripts/pvt_midday.py [--out pvt_paper/run_midday]
Inputs: data/pvt.csv (user grant 2026-11-18; only derived per-test numbers are saved, under R##), foundation/_private_key.csv (handle -> R##),
foundation/linked_v4_pvt.csv, _work/aq_min/<node>.csv, pm_hourly_by_node.csv, nodes.csv.
Clock rule: explicit -7 h for local midnight; no .timestamp() on naive times. Helpers copied from scripts/pvt_added.py.
"""
import argparse, ast, json, math, os
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser(); ap.add_argument("--out", default=os.path.join(ROOT, "pvt_paper", "run_midday"))
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
def loc_midnight_unix(dates):  # naive local date -> Unix of local 00:00, explicit -7 h
    return ((pd.to_datetime(dates) - pd.Timestamp("1970-01-01")).dt.total_seconds() - 7 * 3600).to_numpy()

L = pd.read_csv(os.path.join(ROOT, "foundation", "linked_v4_pvt.csv"))
L["ts"] = pd.to_datetime(L.timestamp_end_local)
D = L[(L.role == "rider") & ~tobool(L.P11_outside_enrolment).fillna(False).astype(bool)].sort_values(["pid", "ts"]).reset_index(drop=True)
D["Y1"] = 100 * (D.y1_speed_rt100 / D.groupby("pid").y1_speed_rt100.transform("mean") - 1)
D["days30"] = D.days_since_enrolment / 30.0; D["logsess"] = np.log(D.session_number)
D["wd0"] = loc_midnight_unix(D.workday)

def fit(df, y, spec, focus, label):           # per-person OLS -> DL + HKSJ (engine as pvt_analysis.py)
    cols = [y] + [c for _, c, _ in spec]; rows = []
    for pid, d in df.dropna(subset=cols).groupby("pid"):
        if len(d) < 15: continue
        X = pd.DataFrame({lab: (zs(d[c]) if tr == "z" else d[c].astype(float)) for lab, c, tr in spec}, index=d.index)
        if any(not (X[f].std() > 1e-9) for f in focus): continue
        X = X[[c for c in X.columns if X[c].std() > 1e-9]]
        Xm = np.column_stack([np.ones(len(d)), X.to_numpy(float)])
        if np.linalg.matrix_rank(Xm) < Xm.shape[1]: continue
        bb, se = ols(d[y].to_numpy(float), Xm, d.workday.to_numpy())
        r = dict(pid=pid, n=len(d))
        for f in focus: j = 1 + list(X.columns).index(f); r["b_" + f] = bb[j]; r["se_" + f] = se[j]
        rows.append(r)
    pr = pd.DataFrame(rows); out = {}
    if len(pr) < 3: RES[label] = None; NM[0] += 1; return {f: None for f in focus}
    for f in focus: p = pool(pr["b_" + f], pr["se_" + f]); p["n"] = int(pr.n.sum()); out[f] = p
    RES[label] = out; NM[0] += 1; return out
def fmt(p, d=2):
    if p is None: return "not estimable (< 3 riders with n ≥ 15)"
    return f"{p['mu']:+.{d}f} [{p['lo_hk']:+.{d}f}, {p['hi_hk']:+.{d}f}], p {p['p_hk']:.3f}; k {p['k']}, n {p['n']}, I² {p['i2']:.0f}%, β<0 in {p['n_neg']}/{p['k']}"
def fe_ols(df, y, xs, fes, label, focus):     # FE by alternating projections; CR1 by rider, t(G-1)
    d = df.dropna(subset=[y] + xs).copy()
    M = d[[y] + xs].to_numpy(float).copy()
    for _ in range(2000):
        prev = M.copy()
        for f in fes: M = M - pd.DataFrame(M, index=d.index).groupby(d[f].to_numpy()).transform("mean").to_numpy()
        if np.abs(M - prev).max() < 1e-10: break
    b, se = ols(M[:, 0], M[:, 1:], d.pid.to_numpy(), min_g=2)
    G = d.pid.nunique(); out = {}
    for f in focus:
        j = xs.index(f); tc = t_crit(G - 1)
        out[f] = dict(mu=b[j], lo=b[j] - tc * se[j], hi=b[j] + tc * se[j], p=p_t(b[j] / se[j], G - 1), n=len(d), G=G)
    RES[label] = out; NM[0] += 1; return out
def f2(r, d=2): return f"{r['mu']:+.{d}f} [{r['lo']:+.{d}f}, {r['hi']:+.{d}f}], p {r['p']:.3f}; n {r['n']}, riders {r['G']}"


# ===================== trial features from data/pvt.csv =====================
PV = pd.read_csv(os.path.join(ROOT, "data", "pvt.csv"))
KEY = pd.read_csv(os.path.join(ROOT, "foundation", "_private_key.csv"))
PV["pid"] = PV.username.astype(str).str.strip().map(dict(zip(KEY.user.astype(str).str.strip(), KEY.pid)))
PV = PV[PV.pid.notna()].copy(); PV["ts"] = pd.to_datetime(PV.timestamp)
def feats(row):
    try:
        rt = np.asarray(ast.literal_eval(row.response_times), float); st = np.asarray(ast.literal_eval(row.stimulation_times), float)
    except Exception: return pd.Series(dtype=float)
    n = min(len(rt), len(st)); rt, st = rt[:n], st[:n]; ok = rt >= 100
    out = {}
    def top(sp):
        if len(sp) < 5: return np.nan
        k = max(1, int(math.ceil(0.1 * len(sp)))); return float(np.sort(sp)[-k:].mean())
    out["F10"] = top(1000 / rt[ok]); out["F10b"] = top(1000 / rt[ok & (st > 60000)])
    for m in range(5):
        sel = ok & (st >= m * 60000) & (st < (m + 1) * 60000); out[f"m{m+1}"] = float((1000 / rt[sel]).mean()) if sel.sum() >= 3 else np.nan
    return pd.Series(out)
TF = pd.concat([PV[["pid", "ts"]], PV.apply(feats, axis=1)], axis=1).drop_duplicates(["pid", "ts"])
TF.to_csv(os.path.join(A.out, "trial_features.csv"), index=False)        # R## + timestamp + numbers only
n0 = len(D); D = D.merge(TF, on=["pid", "ts"], how="left")
P("## Join and features")
P(f"- per-tap features joined to {D.F10.notna().sum()}/{n0} rider tests ({D.F10.notna().mean():.1%})")
for c in ["F10", "F10b"]: D[c + "p"] = 100 * (D[c] / D.groupby("pid")[c].transform("mean") - 1)
D["medp"] = -100 * (D.med_rt / D.groupby("pid").med_rt.transform("mean") - 1)      # sign flipped: + = faster
D["Y3"] = 100 * (D.y3_slowest10_speed / D.groupby("pid").y3_slowest10_speed.transform("mean") - 1)
D["order"] = D.sort_values("ts").groupby(["pid", "workday"]).cumcount() + 1
D["date"] = pd.to_datetime(D.workday); D["mon"] = D.ts.dt.to_period("M").astype(str)
M = D[D.order == 2].copy(); Fst = D[D.order == 1].set_index(["pid", "workday"])
M = M.join(Fst[["Y1", "F10p", "test_end_unix", "node"]].rename(columns=lambda c: c + "_1"), on=["pid", "workday"])
M["dY1"] = M.Y1 - M.Y1_1; M["dF10"] = M.F10p - M.F10p_1; M["gap_h"] = (M.test_start_unix - M.test_end_unix_1) / 3600
P(f"- midday (2nd) tests {len(M)} from {M.pid.nunique()} riders; clock hour median {M.clock_hour.median():.1f} (IQR {M.clock_hour.quantile(.25):.1f}–{M.clock_hour.quantile(.75):.1f}); "
  f"during shift {(M.protocol_slot == 'during_shift').mean():.0%}")
sp = D[[f"m{i}" for i in range(1, 6)]].div(D[[f"m{i}" for i in range(1, 6)]].mean(axis=1), axis=0).mean() * 100 - 100
P("- settling, speed by minute (% of the test's own mean), all tests: " + ", ".join(f"min {i+1} {v:+.1f}" for i, v in enumerate(sp)))
for o_ in [1, 2, 3]:
    z = D[D.order == o_]; P(f"- test {o_} of the day: n {len(z)}, mean speed {z.Y1.mean():+.2f}%, F10 {z.F10p.mean():+.2f}% of own mean, clock median {z.clock_hour.median():.1f}")

# ===================== city reference (as §Q) =====================
NO = pd.read_csv(os.path.join(ROOT, "foundation", "nodes.csv"))[["node", "city"]]
H = pd.read_csv(os.path.join(ROOT, "foundation", "pm_hourly_by_node.csv"))
H = H[(H.node_class == "rider") & (H.minutes_pm >= 30)].merge(NO, on="node")
Hc = H.groupby(["city", "hour_unix"]).agg(med=("pm_mean", "median"), n=("pm_mean", "size"), hl=("hour_local", "first")).reset_index()
Hc = Hc[Hc.n >= 2]; Hc["date"] = pd.to_datetime(Hc.hl.str[:10])
DR = Hc.groupby(["city", "date"]).med.mean().rename("ref").reset_index()
DR["base"] = DR.groupby("city").ref.transform(lambda x: x[(DR.loc[x.index, "date"] >= "2025-12-15") & (DR.loc[x.index, "date"] <= "2026-01-31")].median())
DR["ratio"] = DR.ref / DR.base
DR["haze"] = (DR.ref >= DR.groupby("city").ref.transform(lambda x: x.quantile(2 / 3))).astype(float)
M = M.merge(DR[["city", "date", "ratio", "haze"]], on=["city", "date"], how="left")

# ===================== R-A seasonal table =====================
P("\n## R-A Midday test by month and city (means; % of own mean, + = faster; PM ratio = city reference ÷ its Dec 15–Jan 31 median)")
DR["mon"] = DR.date.dt.to_period("M").astype(str)
pm_m = DR.groupby(["city", "mon"]).ratio.mean()
T_ = M.groupby(["city", "mon"]).agg(n=("Y1", "size"), Y1=("Y1", "mean"), F10=("F10p", "mean"), F10b=("F10bp", "mean"), med=("medp", "mean"), Y3=("Y3", "mean"), d=("dY1", "mean"))
P("| City | Month | n | City PM ratio | Mean speed | Best taps F10 | F10 after 1 min | Median | Slowest 10% | Midday − morning |\n|---|---|---|---|---|---|---|---|---|---|")
for (c_, m_), r_ in T_[T_.n >= 10].iterrows():
    P(f"| {c_} | {m_} | {r_.n:.0f} | ×{pm_m.get((c_, m_), np.nan):.2f} | {r_.Y1:+.1f} | {r_.F10:+.1f} | {r_.F10b:+.1f} | {r_.med:+.1f} | {r_.Y3:+.1f} | {r_.d:+.1f} |")
T_.to_csv(os.path.join(A.out, "monthly_midday.csv"))
pm_m.to_csv(os.path.join(A.out, "monthly_city_pm_ratio.csv"))

ROWS = []
def rec(fam, lab, p):
    if p is None: return
    if "lo_hk" in p: ROWS.append(dict(fam=fam, lab=lab, mu=p["mu"], lo=p["lo_hk"], hi=p["hi_hk"], p=p["p_hk"], n=p["n"], k=p["k"]))
    else: ROWS.append(dict(fam=fam, lab=lab, mu=p["mu"], lo=p["lo"], hi=p["hi"], p=p["p"], n=p["n"], k=p["G"]))
TT = [("days", "days30", "raw"), ("logs", "logsess", "raw")]; CL = [("clock", "clock_hour", "raw")]

# ===================== R-B haze days within city =====================
P("\n## R-B Haze day (top third of the city's days) vs other days, midday test, within rider (two-stage)")
P(f"- haze days: BKK {', '.join(sorted(set(DR[(DR.city=='BKK') & (DR.haze==1)].mon)))}; CMI {', '.join(sorted(set(DR[(DR.city=='CMI') & (DR.haze==1)].mon)))}")
for y, yl in [("Y1", "mean speed"), ("F10p", "best taps F10"), ("F10bp", "F10 after 1 min")]:
    for c_ in ["BKK", "CMI", "both"]:
        s = M if c_ == "both" else M[M.city == c_]
        for adj, spec in [("no trend", CL), ("+ trend", CL + TT)]:
            o = fit(s, y, [("haze", "haze", "raw")] + spec, ["haze"], f"RB_{y}_{c_}_{adj}")["haze"]
            P(f"- {yl}, {c_}, {adj}: {fmt(o)}"); rec("R-B", f"{yl}, {c_}, {adj}", o)

# ===================== R-C personal same-day PM =====================
P("\n## R-C Personal same-day PM (within-rider z) → midday test (two-stage, set B)")
M["PMs"] = M.pm_sameday.where(tobool(M.cov_ok_sameday).fillna(False).astype(bool))
M["PMa"] = M.pm_after.where(tobool(M.cov_ok_after).fillna(False).astype(bool))
B = CL + TT
for y, yl in [("Y1", "mean speed"), ("F10p", "best taps F10"), ("F10bp", "F10 after 1 min"), ("medp", "median speed"), ("Y3", "slowest 10%")]:
    o = fit(M, y, [("pm", "PMs", "z")] + B, ["pm"], f"RC_{y}")["pm"]; P(f"- {yl}: {fmt(o)}"); rec("R-C", yl, o)
for y, yl in [("Y1", "mean speed"), ("F10p", "best taps F10")]:
    o = fit(M, y, [("pm", "PMa", "z")] + B, ["pm"], f"RC_plac_{y}")["pm"]; P(f"- placebo PM after, {yl}: {fmt(o)}"); rec("R-C placebo", yl, o)
    o = fit(M[M.protocol_slot == "during_shift"], y, [("pm", "PMs", "z")] + B, ["pm"], f"RC_shift_{y}")["pm"]; P(f"- during-shift midday tests only, {yl}: {fmt(o)}"); rec("R-C check", yl + " during shift", o)

# ===================== R-D change from the morning test =====================
P("\n## R-D Midday minus morning test (same day) ~ PM between the two tests (z) + gap + T")
M["PMg"] = np.nan
for node, idx in M.groupby("node").groups.items():
    if not np.isfinite(node): continue
    f = os.path.join(ROOT, "foundation", "_work", "aq_min", f"{int(node)}.csv")
    if not os.path.exists(f): continue
    m = pd.read_csv(f).sort_values("minute_unix"); t = m.minute_unix.to_numpy(float)
    C = [np.concatenate([[0], np.cumsum(a)]) for a in (m.n_pm.to_numpy(float), m.pm_sum.to_numpy(float), (m.n_pm.to_numpy() > 0).astype(float))]
    r = M.loc[idx]; a = np.searchsorted(t, r.test_end_unix_1.to_numpy(float), "left"); b = np.searchsorted(t, r.test_start_unix.to_numpy(float), "left")
    n_ = C[0][b] - C[0][a]; mins = C[2][b] - C[2][a]
    M.loc[idx, "PMg"] = np.where((mins >= 30) & (n_ > 0) & (r.node_1.to_numpy() == node), (C[1][b] - C[1][a]) / np.where(n_ > 0, n_, 1), np.nan)
P(f"- pairs with ≥ 30 min between tests: {M.PMg.notna().sum()}; gap median {M.gap_h[M.PMg.notna()].median():.1f} h; midday − morning mean speed {M.dY1.mean():+.2f}%, F10 {M.dF10.mean():+.2f}%")
G = [("gap", "gap_h", "raw")] + TT
for y, yl in [("dY1", "mean speed change"), ("dF10", "best-taps change")]:
    o = fit(M, y, [("pm", "PMg", "z")] + G, ["pm"], f"RD_{y}")["pm"]; P(f"- {yl}: {fmt(o)}"); rec("R-D", yl, o)
    o = fit(M.dropna(subset=["PMg"]), y, [("pm", "PMa", "z")] + G, ["pm"], f"RD_plac_{y}")["pm"]; P(f"- placebo PM after midday, {yl}: {fmt(o)}"); rec("R-D placebo", yl, o)

# ===================== R-E two-way FE (teammates' §Q2) =====================
P("\n## R-E Rider above city, same city and day (log ratio per typical within-rider SD; rider FE + city×date FE; CR1 by rider)")
H["ref_loo"] = H.groupby(["city", "hour_unix"]).pm_mean.transform(lambda x: pd.Series([np.median(np.delete(x.to_numpy(), i)) if len(x) >= 3 else np.nan for i in range(len(x))], index=x.index))
for col in ["q2", "q2a"]: M[col] = np.nan
for node, idx in M.groupby("node").groups.items():
    if not np.isfinite(node): continue
    h = H[(H.node == node) & H.ref_loo.notna()].sort_values("hour_unix")
    if h.empty: continue
    t = h.hour_unix.to_numpy(float); C = [np.concatenate([[0], np.cumsum(a)]) for a in (h.pm_mean.to_numpy(), h.ref_loo.to_numpy(), np.ones(len(h)))]
    r = M.loc[idx]
    for col, lo, hi in [("q2", r.wd0.to_numpy() + 4 * 3600, r.test_start_unix.to_numpy(float) - 3600), ("q2a", r.test_end_unix.to_numpy(float), r.wd0.to_numpy() + 86400 - 3600)]:
        a = np.searchsorted(t, lo, "left"); b = np.searchsorted(t, hi, "right"); n_ = C[2][b] - C[2][a]; ok = n_ >= 2
        M.loc[idx, col] = np.where(ok, np.log(np.where(ok, C[0][b] - C[0][a], 1) / np.where(ok, C[1][b] - C[1][a], 1)), np.nan)
M["cd"] = M.city + "_" + M.workday.astype(str)
for col, tag in [("q2", "before the test"), ("q2a", "placebo, after the test")]:
    for y, yl in [("Y1", "mean speed"), ("F10p", "best taps F10")]:
        s = M.dropna(subset=[col, y, "clock_hour"]).copy(); s = s[s.groupby("cd").pid.transform("nunique") >= 2]
        sw = float(np.sqrt((s[col] - s.groupby("pid")[col].transform("mean")).pow(2).sum() / (len(s) - s.pid.nunique()))); s["x"] = s[col] / sw
        o = fe_ols(s, y, ["x", "clock_hour"], ["pid", "cd"], f"RE_{col}_{y}", ["x"])["x"]
        P(f"- {yl}, {tag}: {f2(o)}; city-days {s.cd.nunique()}"); rec("R-E" + (" placebo" if col == "q2a" else ""), yl + ", " + tag, o)

# ===================== multiplicity =====================
R_ = pd.DataFrame(ROWS); m = len(R_); order = np.argsort(R_.p.to_numpy()); q = np.empty(m); prev = 1.0
for rank, i in list(enumerate(order, 1))[::-1]: prev = min(prev, R_.p.iloc[i] * m / rank); q[i] = prev
R_["q_BH"] = q; R_.to_csv(os.path.join(A.out, "screen_table.csv"), index=False)
P(f"\n## Screen summary: {m} inferential rows; p < 0.05 in {(R_.p < 0.05).sum()}; BH q < 0.10 in {(R_.q_BH < 0.10).sum()}")
for _, r_ in R_.sort_values("p").head(8).iterrows(): P(f"- {r_.fam} | {r_.lab}: {r_.mu:+.2f} [{r_.lo:+.2f}, {r_.hi:+.2f}], p {r_.p:.3f}, q {r_.q_BH:.2f}")

# ===================== R-F figure =====================
def fig():
    W, Hh = 760, 330; pal = {"F10": "#2a78d6", "Y1": "#eb6834"}; mons = ["2025-12", "2026-01", "2026-02", "2026-03", "2026-04"]; lab = ["Dec", "Jan", "Feb", "Mar", "Apr"]
    sv = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {Hh}" font-family="sans-serif" font-size="12"><rect width="{W}" height="{Hh}" fill="#fcfcfb"/>']
    for pi, c_ in enumerate(["BKK", "CMI"]):
        x0 = 60 + pi * 370; y0, y1 = 50, 250; xs = [x0 + 30 + i * 70 for i in range(5)]
        sy = lambda v: y0 + (10 - v) / 20 * (y1 - y0)                   # -10..+10 %
        mx = max(pm_m[c_].max(), 1); sr = lambda v: y1 - (v / mx) * (y1 - y0)
        sv.append(f'<text x="{x0}" y="30" fill="#0b0b0b" font-weight="bold">{"Bangkok" if c_=="BKK" else "Chiang Mai"}: midday test vs city PM</text>')
        sv.append(f'<line x1="{x0}" x2="{x0+320}" y1="{sy(0)}" y2="{sy(0)}" stroke="#52514e" stroke-dasharray="3,3"/>')
        for v in [-10, -5, 0, 5, 10]: sv.append(f'<text x="{x0-6}" y="{sy(v)+4}" text-anchor="end" fill="#52514e">{v:+d}%</text>')
        bw = 26
        for i, m_ in enumerate(mons):
            v = pm_m.get((c_, m_), np.nan)
            if np.isfinite(v): sv.append(f'<rect x="{xs[i]-bw/2}" y="{sr(v)}" width="{bw}" height="{y1-sr(v)}" fill="#52514e" opacity="0.15"/><text x="{xs[i]}" y="{sr(v)-4}" text-anchor="middle" fill="#52514e" font-size="10">×{v:.1f}</text>')
            sv.append(f'<text x="{xs[i]}" y="{y1+18}" text-anchor="middle" fill="#52514e">{lab[i]}</text>')
        for k_, col in [("F10", "F10"), ("Y1", "Y1")]:
            pts = [(xs[i], sy(T_.loc[(c_, m_), col])) for i, m_ in enumerate(mons) if (c_, m_) in T_.index and T_.loc[(c_, m_), "n"] >= 10]
            sv.append('<polyline fill="none" stroke="{}" stroke-width="2.5" points="{}"/>'.format(pal[k_], " ".join(f"{a:.0f},{b:.0f}" for a, b in pts)))
            sv += [f'<circle cx="{a:.0f}" cy="{b:.0f}" r="3.5" fill="{pal[k_]}"/>' for a, b in pts]
    sv.append(f'<text x="60" y="{Hh-20}" fill="#2a78d6">● best taps (fastest 10%)</text><text x="250" y="{Hh-20}" fill="#eb6834">● mean speed</text>'
              f'<text x="400" y="{Hh-20}" fill="#52514e">bars: city PM ÷ own Dec–Jan median (raw sensors)</text><text x="60" y="{Hh-5}" fill="#52514e" font-size="10">% of own mean; + = faster. 2nd test of the workday.</text></svg>')
    open(os.path.join(A.out, "fig_midday_trajectory.svg"), "w").write("\n".join(sv))
fig()
P(f"\n## Model count: {NM[0]}")
open(os.path.join(A.out, "results_midday.md"), "w").write("\n".join(OUT) + "\n")
print("\n".join(OUT))
