"""SPEC_ADDENDUM §Q: added analyses Q4 (previous night's home air), Q2 (personal exposure above the city, two-way FE),
Q3 (haze onset DiD). Written 2026-11-17 before computing; run once; ALL EXPLORATORY.
Usage: python3 -I scripts/pvt_added.py [--out pvt_paper/run_added]
Inputs (grant G3): foundation/linked_v4_pvt.csv, _work/aq_min/<node>.csv, _work/home_stays.csv, pm_hourly_by_node.csv, nodes.csv.
Clock rule: local midnight -> Unix with an explicit -7 h; no .timestamp() on naive times.
Stats helpers copied unchanged from scripts/pvt_heat.py.
"""
import argparse, json, math, os
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser(); ap.add_argument("--out", default=os.path.join(ROOT, "pvt_paper", "run_added"))
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

# ===================== Q4 previous night's home air =====================
HS = pd.read_csv(os.path.join(ROOT, "foundation", "_work", "home_stays.csv"))
F = D.groupby(["pid", "workday"], as_index=False).head(1).copy()
for c in ["night", "night_n", "next", "next_n"]: F[c] = np.nan
for (pid, node), idx in F.groupby(["pid", "node"]).groups.items():
    if not np.isfinite(node): continue
    fpath = os.path.join(ROOT, "foundation", "_work", "aq_min", f"{int(node)}.csv")
    if not os.path.exists(fpath): continue
    m = pd.read_csv(fpath).sort_values("minute_unix"); t = m.minute_unix.to_numpy(float)
    st = HS[(HS.pid == pid) & (HS.sensor == int(node))].sort_values("start")
    home = np.zeros(len(t), bool)
    for s0, s1 in zip(st.start.to_numpy(float), st.end.to_numpy(float)):
        home[np.searchsorted(t, s0, "left"):np.searchsorted(t, s1, "left")] = True
    npm = m.n_pm.to_numpy(float) * home; spm = m.pm_sum.to_numpy(float) * home; cnt = ((m.n_pm.to_numpy() > 0) & home).astype(float)
    C = [np.concatenate([[0], np.cumsum(a)]) for a in (npm, spm, cnt)]
    rows = F.loc[idx]
    for tag, lo, hi in [("night", rows.wd0.to_numpy(), np.minimum(rows.wd0.to_numpy() + 5 * 3600, rows.test_start_unix.to_numpy(float))),
                        ("next", rows.wd0.to_numpy() + 86400, rows.wd0.to_numpy() + 86400 + 5 * 3600)]:
        a = np.searchsorted(t, lo, "left"); b = np.searchsorted(t, hi, "left")
        n_ = C[0][b] - C[0][a]; mins = C[2][b] - C[2][a]
        F.loc[idx, tag] = np.where(mins >= 120, (C[1][b] - C[1][a]) / np.where(n_ > 0, n_, 1), np.nan)
        F.loc[idx, tag + "_n"] = mins
P("## Q4 Previous night's home air (00:00–05:00, home-stay minutes, ≥ 120 min) → first test of the workday")
P(f"- first tests {len(F)}; with night exposure {F.night.notna().sum()} ({F.night.notna().mean():.0%}); with next-night (placebo) {F.next.notna().sum()}; "
  f"home minutes median {F.night_n[F.night.notna()].median():.0f}; first test before 12:00: {(F.clock_hour < 12).mean():.0%}")
B = [("clock", "clock_hour", "raw"), ("days", "days30", "raw"), ("logs", "logsess", "raw")]
o = fit(F, "Y1", [("night", "night", "z")] + B, ["night"], "Q4_primary"); P(f"- **primary, per +1 within-rider SD: {fmt(o['night'])}**")
o = fit(F[F.clock_hour < 12], "Y1", [("night", "night", "z")] + B, ["night"], "Q4_before12"); P(f"- first test before 12:00: {fmt(o['night'])}")
Fz = F.copy(); Fz["night_zz"] = Fz.groupby("pid").night.transform(zs)
o = fe_ols(Fz, "Y1", ["night_zz", "days30", "clock_hour"], ["pid"], "Q4_fe", ["night_zz"]); P(f"- sensitivity, rider FE + days/30 + clock hour: {f2(o['night_zz'])}")
P(f"- following-night placebo available for {F.next.notna().sum()} first tests in {F[F.next.notna()].pid.nunique()} riders (AQ export covers study days only)")
F = F.sort_values(["pid", "ts"]); F["nextwd"] = F.groupby("pid").night.shift(-1)   # deviation (2026-11-17, before results): night before the rider's next workday
o = fit(F.dropna(subset=["night", "nextwd"]), "Y1", [("nw", "nextwd", "z")] + B, ["nw"], "Q4_placebo_nextwd")
P(f"- **placebo (deviation), night before the rider's next workday: {fmt(o['nw'])}**")
o = fit(F.dropna(subset=["night", "nextwd"]), "Y1", [("night", "night", "z")] + B, ["night"], "Q4_primary_same_nw")
P(f"- primary on the same sessions: {fmt(o['night'])}")
s = F.dropna(subset=["night", "next"])
o = fit(s, "Y1", [("night", "night", "z")] + B, ["night"], "Q4_primary_same"); P(f"- primary on sessions that also have the placebo: {fmt(o['night'])}")
o = fit(s, "Y1", [("next", "next", "z")] + B, ["next"], "Q4_placebo"); P(f"- **placebo, following night: {fmt(o['next'])}**")
s = F.dropna(subset=["night", "nextwd"]); w = s.assign(a=s.night - s.groupby("pid").night.transform("mean"), b=s.nextwd - s.groupby("pid").nextwd.transform("mean"))
P(f"- within-rider r(night, night before next workday) {np.corrcoef(w.a, w.b)[0,1]:+.2f}")

# ===================== hourly city reference =====================
NO = pd.read_csv(os.path.join(ROOT, "foundation", "nodes.csv"))[["node", "city"]]
H = pd.read_csv(os.path.join(ROOT, "foundation", "pm_hourly_by_node.csv"))
H = H[(H.node_class == "rider") & (H.minutes_pm >= 30)].merge(NO, on="node")
g = H.groupby(["city", "hour_unix"]).pm_mean
H["n_city"] = g.transform("size")
def loo_med(x):
    v = x.to_numpy(); return pd.Series([np.median(np.delete(v, i)) if len(v) >= 3 else np.nan for i in range(len(v))], index=x.index)
H["ref_loo"] = g.transform(loo_med)

# ===================== Q2 personal above city, two-way FE =====================
D["q2"] = np.nan; D["q2_after"] = np.nan; D["q2_h"] = 0
for node, idx in D.groupby("node").groups.items():
    if not np.isfinite(node): continue
    h = H[(H.node == node) & H.ref_loo.notna()].sort_values("hour_unix")
    if h.empty: continue
    t = h.hour_unix.to_numpy(float); C = [np.concatenate([[0], np.cumsum(a)]) for a in (h.pm_mean.to_numpy(), h.ref_loo.to_numpy(), np.ones(len(h)))]
    r = D.loc[idx]
    for col, lo, hi in [("q2", r.wd0.to_numpy() + 4 * 3600, r.test_start_unix.to_numpy(float) - 3600),       # hours fully before test start
                        ("q2_after", r.test_end_unix.to_numpy(float), r.wd0.to_numpy() + 86400 - 3600)]:       # hours fully after test end, to 24:00
        a = np.searchsorted(t, lo, "left"); b = np.searchsorted(t, hi, "right"); n_ = C[2][b] - C[2][a]
        ok = n_ >= 2
        D.loc[idx, col] = np.where(ok, np.log(np.where(ok, (C[0][b] - C[0][a]), 1) / np.where(ok, (C[1][b] - C[1][a]), 1)), np.nan)
        if col == "q2": D.loc[idx, "q2_h"] = n_
P("\n## Q2 Personal exposure above the city (log rider ÷ leave-one-out city reference; rider FE + city×date FE; CR1 by rider)")
D["cd"] = D.city + "_" + D.workday.astype(str)
def q2(col, label):
    s = D.dropna(subset=[col, "clock_hour"]).copy()
    s = s[s.groupby("cd").pid.transform("nunique") >= 2]
    sw = float(np.sqrt((s[col] - s.groupby("pid")[col].transform("mean")).pow(2).sum() / (len(s) - s.pid.nunique())))
    s[col + "_s"] = s[col] / sw
    o = fe_ols(s, "Y1", [col + "_s", "clock_hour"], ["pid", "cd"], label, [col + "_s"])[col + "_s"]
    return o, sw, s
o, sw, s = q2("q2", "Q2_main")
P(f"- sample: {len(s)} tests, {s.cd.nunique()} city-days with ≥ 2 riders; pre-test hours median {s.q2_h.median():.0f}; typical within-rider SD of the log ratio {sw:.2f} "
  f"(×{math.exp(sw):.2f}); rider above city on {(s.q2 > 0).mean():.0%} of tests")
P(f"- **main, % per typical within-rider SD: {f2(o)}**")
o, sw2, s2 = q2("q2_after", "Q2_placebo"); P(f"- **placebo (hours after the test): {f2(o)}** (n {len(s2)})")
w = D.dropna(subset=["q2", "q2_after"]); P(f"- within-rider r(before, after) {np.corrcoef(w.q2 - w.groupby('pid').q2.transform('mean'), w.q2_after - w.groupby('pid').q2_after.transform('mean'))[0,1]:+.2f}")

# ===================== Q3 haze onset DiD =====================
Hc = H.groupby(["city", "hour_unix"]).agg(med=("pm_mean", "median"), n=("pm_mean", "size"), hl=("hour_local", "first")).reset_index()
Hc = Hc[Hc.n >= 2]; Hc["date"] = pd.to_datetime(Hc.hl.str[:10])
DR = Hc.groupby(["city", "date"]).med.mean().unstack(0)
c_ = DR.CMI.dropna()                                   # study days only (AQ export covers ~2-3 days/week later on)
roll = c_.rolling("7D", min_periods=2).mean()          # deviation 2026-11-17 (before outcomes): mean of available study days in the trailing 7 calendar days
base = c_[(c_.index >= "2025-12-15") & (c_.index <= "2026-01-31")].median()
cand = roll[(roll.index >= "2026-02-01") & (roll >= 2 * base)]
ONSET = cand.index.min()
sus = [d for d in cand.index if (roll[(roll.index >= d) & (roll.index < d + pd.Timedelta(days=14))] >= 2 * base).all()]
ONSET2 = min(sus)
P("\n## Q3 Haze onset (CMI), difference-in-differences")
P(f"- **onset (PM only, computed before any outcome): {ONSET.date()}** (trailing 7-day mean ÷ Dec 15–Jan 31 median = {roll[ONSET]/base:.2f}×; {len(c_[(c_.index >= '2025-12-15') & (c_.index <= '2026-01-31')])} baseline study days). "
  f"Sensitivity onset (first date that stays ≥ 2× for 14 days): **{ONSET2.date()}**. S4's §15 date: not yet written (check when available).")
D["date"] = pd.to_datetime(D.workday)
for ONSET, W, lab in [(ONSET, 14, "Q3_pm14"), (ONSET, 21, "Q3_pm21"), (ONSET2, 14, "Q3_sustained_pm14")]:
    P(f"- onset {ONSET.date()}:")
    s = D[(D.date >= ONSET - pd.Timedelta(days=W)) & (D.date < ONSET + pd.Timedelta(days=W))].copy()
    s["post"] = (s.date >= ONSET).astype(float); s["cmi"] = (s.city == "CMI").astype(float); s["did"] = s.post * s.cmi
    o = fe_ols(s, "Y1", ["did", "clock_hour"], ["pid", "date"], lab, ["did"])["did"]
    P(f"- **±{W} d: post × CMI {f2(o)}** (date FE absorbs the common season slowing, if both cities slow alike)")
    for c in ["BKK", "CMI"]:
        z = s[s.city == c]; pre = z[z.post == 0]; post = z[z.post == 1]
        rr = DR[c].dropna()[(DR[c].dropna().index >= ONSET - pd.Timedelta(days=W)) & (DR[c].dropna().index < ONSET + pd.Timedelta(days=W))]
        ratio = rr[rr.index >= ONSET].mean() / rr[rr.index < ONSET].mean()
        P(f"  - {c}: Y1 {pre.Y1.mean():+.2f}% (n {len(pre)}, {pre.pid.nunique()} riders, {pre.date.nunique()} days) → {post.Y1.mean():+.2f}% (n {len(post)}, {post.date.nunique()} days); "
          f"city PM after ÷ before ×{ratio:.2f}")
P(f"\n## Model count: {NM[0]}")
open(os.path.join(A.out, "results_added.md"), "w").write("\n".join(OUT) + "\n")
json.dump(RES, open(os.path.join(A.out, "pooled_added.json"), "w"), indent=1, default=float)
print("\n".join(OUT))
