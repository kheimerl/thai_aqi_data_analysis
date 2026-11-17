"""SPEC_ADDENDUM §C: heat as a named rival (EXPLORATORY), approved by the user 2026-11-12. Run once.

Usage: python3 -I scripts/pvt_heat.py [--pm-col pm] [--out pvt_paper/run_heat]
Inputs: foundation/linked_v4_pvt.csv; data/review_2026_09/metar_VTBD.csv, metar_VTCC.csv (scripts/pvt_fetch_metar.py).
Ambient = airport METAR air temperature nearest the test start (Bangkok local on both sides; +-90 min), VTBD for BKK
riders, VTCC for CMI riders. Onboard temp_* is NOT used. Same sample, outcome (Y1 = RT >= 100 ms), time term T,
covariate set B, per-rider OLS (CR1 by workday if >= 20 workdays, else HC1) and DL + HKSJ pooling as
scripts/pvt_analysis.py (helpers copied, unchanged).
"""
import argparse, json, math, os
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser()
ap.add_argument("--pm-col", default="pm")
ap.add_argument("--out", default=os.path.join(ROOT, "pvt_paper", "run_heat"))
A = ap.parse_args(); os.makedirs(A.out, exist_ok=True); PM = A.pm_col

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

# ---------- data (as pvt_analysis.py) ----------
L = pd.read_csv(os.path.join(ROOT, "foundation", "linked_v4_pvt.csv"))
D = L[(L.role == "rider") & (~tobool(L.P11_outside_enrolment).fillna(False).astype(bool))].copy()
D["ts"] = pd.to_datetime(D.timestamp_end_local)
D = D.sort_values(["pid", "ts"]).reset_index(drop=True)
D["late_starter_handover"] = tobool(D.late_starter_handover)
D["Y1"] = 100 * (D.y1_speed_rt100 / D.groupby("pid").y1_speed_rt100.transform("mean") - 1)
D["days30"] = D.days_since_enrolment / 30.0
D["logsess"] = np.log(D.session_number)
D["hos"] = D.hours_on_shift_before_test
D["PMs"] = D[f"{PM}_sameday"].where(tobool(D.cov_ok_sameday).fillna(False).astype(bool))

# ---------- ambient match (Bangkok local both sides; nearest obs within +-90 min) ----------
D["t_start"] = D.ts - pd.Timedelta(seconds=300)
OUT = []
def P(s=""): OUT.append(s)
P("## C diagnostics (before any model)")
for city, st in [("BKK", "VTBD"), ("CMI", "VTCC")]:
    m = pd.read_csv(os.path.join(ROOT, "data", "review_2026_09", f"metar_{st}.csv"), na_values="M").dropna(subset=["tmpc"])
    m["v"] = pd.to_datetime(m.valid); m = m.sort_values("v")[["v", "tmpc"]]
    P(f"- {st}: {len(m)} obs {m.v.min()} – {m.v.max()}; median °C by local hour: " +
      ", ".join(f"{h}:{t:.0f}" for h, t in m.groupby(m.v.dt.hour).tmpc.median().items()))
    for col, shift in [("amb", 0), ("amb4", 4)]:
        idx = D.city == city
        q = D.loc[idx, ["t_start"]].assign(key=D.loc[idx, "t_start"] + pd.Timedelta(hours=shift), row=D.index[idx]).sort_values("key")
        # merge_asof returns a fresh RangeIndex: write back by the saved row label (fix 2026-11-12, see STATUS)
        mm = pd.merge_asof(q, m.rename(columns={"v": "key"}), on="key", direction="nearest", tolerance=pd.Timedelta(minutes=90))
        D.loc[mm["row"].to_numpy(), col] = mm.tmpc.to_numpy()
for city, g in D.groupby("city"):
    P(f"- coverage {city}: amb {g.amb.notna().mean():.1%}, amb+4h {g.amb4.notna().mean():.1%} of {len(g)} tests")
Z = D.dropna(subset=["amb", "clock_hour", "days30"]).copy()
for c in ["amb", "clock_hour", "days30", "logsess"]: Z[c + "_w"] = Z[c] - Z.groupby("pid")[c].transform("mean")
r_ch = np.corrcoef(Z.amb_w, Z.clock_hour_w)[0, 1]; r_d = np.corrcoef(Z.amb_w, Z.days30_w)[0, 1]
X = np.column_stack([np.ones(len(Z)), Z.clock_hour_w, Z.days30_w, Z.logsess_w]); b = np.linalg.lstsq(X, Z.amb_w, rcond=None)[0]
r2 = 1 - ((Z.amb_w - X @ b) ** 2).sum() / ((Z.amb_w - Z.amb_w.mean()) ** 2).sum()
P(f"- within-rider r(ambient, clock hour) {r_ch:+.2f}; r(ambient, days since enrolment) {r_d:+.2f}; "
  f"R² of ambient on B (clock, days, log session) {r2:.2f} → VIF {1/(1-r2):.1f}")
P(f"- within-rider SD of ambient °C: median {D.groupby('pid').amb.std().median():.2f} °C (range {D.groupby('pid').amb.std().min():.2f}–{D.groupby('pid').amb.std().max():.2f})")

# ---------- per-rider engine ----------
NM = [0]; RES = {}; PER = {}
def fit(df, spec, focus, label):
    cols = ["Y1"] + [c for _, c, _ in spec]
    rows = []
    for pid, d in df.dropna(subset=cols).groupby("pid"):
        if len(d) < 15: continue
        Xd = {}
        for lab, c, tr in spec: Xd[lab] = zs(d[c]) if tr == "z" else d[c].astype(float)
        X = pd.DataFrame(Xd, index=d.index)
        if any(X[f].std() < 1e-9 or not np.isfinite(X[f]).all() for f in focus): continue
        X = X[[c for c in X.columns if X[c].std() > 1e-9]]
        Xm = np.column_stack([np.ones(len(d)), X.to_numpy(float)])
        if np.linalg.matrix_rank(Xm) < Xm.shape[1]: continue
        bb, se = ols(d.Y1.to_numpy(float), Xm, d.workday.to_numpy())
        r = dict(pid=pid, city=d.city.iloc[0], n=len(d), sd_amb=d.amb.std(ddof=1))
        for f in focus:
            j = 1 + list(X.columns).index(f); r["b_" + f] = bb[j]; r["se_" + f] = se[j]
        rows.append(r)
    pr = pd.DataFrame(rows); PER[label] = pr; out = {}
    for f in focus:
        p = pool(pr["b_" + f], pr["se_" + f]); p["n"] = int(pr.n.sum()); out[f] = p
    RES[label] = out; NM[0] += 1
    return out
def fmt(p, d=2):
    return (f"{p['mu']:+.{d}f} [{p['lo_hk']:+.{d}f}, {p['hi_hk']:+.{d}f}], p_HKSJ {p['p_hk']:.3f} (DL p {p['p_dl']:.3f}); "
            f"k {p['k']}, n {p['n']}, I² {p['i2']:.0f}%, β<0 in {p['n_neg']}/{p['k']}")
def per_degC(label, f="amb"):
    pr = PER[label]; p = pool(pr["b_" + f] / pr.sd_amb, pr["se_" + f] / pr.sd_amb)
    ms = 1000 / (1000 / MEDRT * (1 + p["mu"] / 100)) - MEDRT
    return f"per 1 °C: {p['mu']:+.3f}% [{p['lo_hk']:+.3f}, {p['hi_hk']:+.3f}] (≈ {ms:+.1f} ms at median RT)"
MEDRT = D.med_rt.median()
T = [("days30", "days30", "raw"), ("logsess", "logsess", "raw")]
B = T + [("clock", "clock_hour", "raw")]

def block(df, tag):
    # C-a: F1-Y1 with/without ambient, same sessions
    s = df.dropna(subset=["PMs", "amb"])
    a0 = fit(s, [("pm", "PMs", "z")] + B, ["pm"], f"Ca0{tag}")
    a1 = fit(s, [("pm", "PMs", "z"), ("amb", "amb", "z")] + B, ["pm", "amb"], f"Ca{tag}")
    P(f"- C-a{tag} F1-Y1 without ambient (same sessions): PM {fmt(a0['pm'])}")
    P(f"- C-a{tag} F1-Y1 + ambient: PM {fmt(a1['pm'])}")
    P(f"- C-a{tag} F1-Y1 + ambient: **ambient per +1 SD {fmt(a1['amb'])}**; {per_degC(f'Ca{tag}')}")
    # C-b: F2a with/without ambient, same sessions
    s = df.dropna(subset=["hos", "amb"])
    b0 = fit(s, [("hos", "hos", "raw")] + T, ["hos"], f"Cb0{tag}")
    b1 = fit(s, [("hos", "hos", "raw"), ("amb", "amb", "z")] + T, ["hos", "amb"], f"Cb{tag}")
    P(f"- C-b{tag} F2a without ambient (same sessions): hours {fmt(b0['hos'], 3)}")
    P(f"- C-b{tag} F2a + ambient: hours {fmt(b1['hos'], 3)}; shift of hours toward 0: {1 - abs(b1['hos']['mu']) / abs(b0['hos']['mu']):.0%}")
    P(f"- C-b{tag} F2a + ambient: **ambient per +1 SD {fmt(b1['amb'])}**; {per_degC(f'Cb{tag}')}")
    return a0, a1, b0, b1

P("\n## C-a / C-b primary (22 riders)")
a0, a1, b0, b1 = block(D, "")
P("\n## C-p placebo: ambient 4 h after the test start")
s = D.dropna(subset=["amb", "amb4"])
p1 = fit(s, [("amb4", "amb4", "z")] + B, ["amb4"], "Cp_alone")
p2 = fit(s, [("amb", "amb", "z"), ("amb4", "amb4", "z")] + B, ["amb", "amb4"], "Cp_joint")
P(f"- C-p alone: ambient +4 h {fmt(p1['amb4'])}")
P(f"- C-p joint: ambient at test {fmt(p2['amb'])}; ambient +4 h {fmt(p2['amb4'])}")
P("\n## C-20 (20-slot sensitivity: without R12, R14)")
D20 = D[D.late_starter_handover != True]
s = D20.dropna(subset=["PMs", "amb"])
c20a = fit(s, [("pm", "PMs", "z"), ("amb", "amb", "z")] + B, ["pm", "amb"], "C20a")
s = D20.dropna(subset=["hos", "amb"])
c20b = fit(s, [("hos", "hos", "raw"), ("amb", "amb", "z")] + T, ["hos", "amb"], "C20b")
P(f"- C-20a F1-Y1 + ambient: PM {fmt(c20a['pm'])}; ambient {fmt(c20a['amb'])}")
P(f"- C-20b F2a + ambient: hours {fmt(c20b['hos'], 3)}; ambient {fmt(c20b['amb'])}")

P("\n## Reading rules (SPEC §C.4; exploratory, no Gate 2 dependence)")
heat = (a1["amb"]["hi_hk"] < 0 or a1["amb"]["lo_hk"] > 0) and (p2["amb4"]["lo_hk"] < 0 < p2["amb4"]["hi_hk"])
P(f"- Heat associated with vigilance (C-a ambient CI excludes 0 AND joint placebo CI includes 0): **{'yes' if heat else 'no'}**")
P(f"- PM bound robust to heat (C-a PM lower limit ≥ −1.01%): **{'yes' if a1['pm']['lo_hk'] >= -1.01 else 'no'}** (lower limit {a1['pm']['lo_hk']:+.2f}%)")
P(f"- Heat carries part of the fatigue gradient (hours moves ≥ 50% toward 0): **{'yes' if 1 - abs(b1['hos']['mu']) / abs(b0['hos']['mu']) >= 0.5 else 'no'}**")
P(f"\nStage-2 pooled models this script: {NM[0]} (inferential tests per spec: 2 + 1 placebo). Label: EXPLORATORY.")

open(os.path.join(A.out, "results_heat.md"), "w").write(f"# §C heat run (--pm-col {PM}); auto-generated by scripts/pvt_heat.py\n\n" + "\n".join(OUT) + "\n")
pd.concat([pr.assign(model=k) for k, pr in PER.items()]).to_csv(os.path.join(A.out, "per_rider_heat.csv"), index=False)
json.dump(dict(pooled=RES, nmodels=NM[0]), open(os.path.join(A.out, "pooled_heat.json"), "w"), indent=1)
print("\n".join(OUT))
