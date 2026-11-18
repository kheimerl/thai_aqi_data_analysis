"""SPEC_ADDENDUM §P3b deferred rows (pre-specified 2026-11-17; run 2026-11-18 after the user's data/pvt.csv grant).
Fastest-10% speed (F10) and within-test decrement (minute 5 vs minute 1 speed) per 30 days, all rider tests, + clock hour.
Reads pvt_paper/run_midday/trial_features.csv (derived by scripts/pvt_midday.py). EXPLORATORY.
"""
import argparse, json, math, os
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
class A_: out = os.path.join(ROOT, "pvt_paper", "run_midday")
A = A_()
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



TF = pd.read_csv(os.path.join(A.out, "trial_features.csv")); TF["ts"] = pd.to_datetime(TF.ts)
D = D.merge(TF, on=["pid", "ts"], how="left")
D["F10p"] = 100 * (D.F10 / D.groupby("pid").F10.transform("mean") - 1)
D["dec"] = 100 * (D.m5 / D.m1 - 1)            # within-test change, minute 5 vs minute 1 (% ; - = slower at the end)
D["dec4"] = 100 * (D[["m4", "m5"]].mean(axis=1) / D.m1 - 1)
P("## P3b deferred rows (per 30 days, + clock hour; two-stage)")
for y, l in [("F10p", "fastest 10% speed, % of own mean"), ("dec", "minute 5 vs minute 1 speed, %"), ("dec4", "minutes 4-5 vs minute 1 speed, %")]:
    o = fit(D, y, [("days", "days30", "raw"), ("clock", "clock_hour", "raw")], ["days"], "P3b_" + y)["days"]
    P(f"- {l} [mean {D[y].mean():+.2f}]: {fmt(o, 3)}")
open(os.path.join(ROOT, "pvt_paper", "run_review", "p3b_deferred.md"), "w").write("\n".join(OUT) + "\n")
print("\n".join(OUT))
