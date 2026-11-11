"""S3 Paper 1 (PVT) analysis, run once per pvt_paper/SPEC_ADDENDUM.md (approved 2026-11-10).

Usage:  python3 -I scripts/pvt_analysis.py [--pm-col pm] [--out pvt_paper/run_raw]
  --pm-col  prefix of the PM columns: pm -> pm_sameday, pm_after, ...; pm_calibrated -> pm_calibrated_sameday, ...
            (columns that do not exist for the prefix are skipped and logged).
numpy/pandas only. Riders only, P11 dropped. Per-rider OLS (cluster by workday if >= 20 workdays, else HC1)
-> DerSimonian-Laird + HKSJ (IntHout floor). All PM exposures are within-rider transformed (z, IQR, tertiles).
"""
import argparse, json, math, os, sys
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser()
ap.add_argument("--pm-col", default="pm")
ap.add_argument("--out", default=os.path.join(ROOT, "pvt_paper", "run_raw"))
A = ap.parse_args()
os.makedirs(A.out, exist_ok=True)
PM = A.pm_col
RNG_SEED = 20261110

# ---------------- statistics (as scripts/analysis_health_v3.py) ----------------
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

def p_t(t, df): return betainc(df / 2, 0.5, df / (df + t * t))  # two-sided
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
    var_hk = (ws * (b - mu) ** 2).sum() / ((k - 1) * ws.sum())
    se_hk = math.sqrt(max(var_hk, 1 / ws.sum()))
    tc = t_crit(k - 1, q); zc = 1.959964 if q == 0.975 else 1.644854
    return dict(k=k, mu=mu, lo_hk=mu - tc * se_hk, hi_hk=mu + tc * se_hk, p_hk=p_t(mu / se_hk, k - 1),
                se_hk=se_hk, se_dl=se_dl, lo_dl=mu - zc * se_dl, hi_dl=mu + zc * se_dl, p_dl=p_z(mu / se_dl),
                tau2=tau2, i2=max(0.0, (Q - (k - 1)) / Q) * 100 if Q > 0 else 0.0,
                n_neg=int((b < 0).sum()), n_pos=int((b > 0).sum()))

# ---------------- data ----------------
def tobool(s):
    return s.map(lambda v: True if v in (True, "True", 1, "1") else (False if v in (False, "False", 0, "0") else np.nan))

L = pd.read_csv(os.path.join(ROOT, "foundation", "linked_v4_pvt.csv"))
D = L[(L.role == "rider") & (~tobool(L.P11_outside_enrolment).fillna(False).astype(bool))].copy()
D["ts"] = pd.to_datetime(D.timestamp_end_local)
D = D.sort_values(["pid", "ts"]).reset_index(drop=True)
for c in ["P8_moving", "test_before_shift", "any_exclusion", "late_starter_handover"]:
    D[c] = tobool(D[c])
# outcomes (rider means over the primary sample)
D["Y1"] = 100 * (D.y1_speed_rtgt0 / D.groupby("pid").y1_speed_rtgt0.transform("mean") - 1)
D["Y2"] = D.y2_lapse_pct_2xmed
D["Y3"] = 100 * (D.y3_slowest10_speed / D.groupby("pid").y3_slowest10_speed.transform("mean") - 1)
# time term T and covariates
D["days30"] = D.days_since_enrolment / 30.0
D["logsess"] = np.log(D.session_number)
D["hos"] = D.hours_on_shift_before_test
D["km10"] = D.km_ridden_before_test / 10.0
D["month"] = pd.to_datetime(D.workday).dt.month
D["start_h"] = (D.clock_hour)  # test start clock hour (decimal)
def band(h):
    h = h % 24
    return "04-10" if 4 <= h < 10 else "10-14" if h < 14 and h >= 10 else "14-18" if 14 <= h < 18 else "18-22" if 18 <= h < 22 else "22-04"
D["band"] = D.clock_hour.map(band)

LOG = []
def col_or_none(c):
    if c in D.columns: return c
    LOG.append(f"column {c} missing for --pm-col {PM}: models using it skipped"); return None

# exposure columns with coverage rules (SPEC §0 and S/H)
def expo(name):
    m = {"sameday": (f"{PM}_sameday", "cov_ok_sameday"), "after": (f"{PM}_after", "cov_ok_after"),
         "onshift": (f"{PM}_onshift", "cov_ok_onshift"),
         "median": (f"{PM}_sameday_median", "cov_ok_sameday"), "clip": (f"{PM}_sameday_clip", "cov_ok_sameday"),
         "peak5": (f"{PM}_sameday_peak5", "cov_ok_sameday"), "cum": (f"{PM}_sameday_cum", "cov_ok_sameday"),
         "preshift": (f"{PM}_preshift", ("pm_preshift_nmin", 30)),
         "after_onshift": (f"{PM}_after_onshift", ("pm_after_onshift_nmin", 30)),
         "after_home": (f"{PM}_after_home", ("pm_after_home_nmin", 30))}[name]
    c, cov = m
    if c not in D.columns:
        LOG.append(f"exposure {c} missing: skipped"); return None
    out = f"E_{name}"
    if isinstance(cov, tuple):
        ok = D[cov[0]] >= cov[1]
    else:
        ok = tobool(D[cov]).fillna(False).astype(bool)
    D[out] = D[c].where(ok)
    D[out + "_nocov"] = D[c]
    return out

EX = {n: expo(n) for n in ["sameday", "after", "onshift", "median", "clip", "peak5", "cum", "preshift", "after_onshift", "after_home"]}
D["temp_h"] = D.temp_sameday  # housing temperature, sensitivity only

# ---------------- per-rider engine ----------------
def zs(x):
    sd = x.std(ddof=1)
    return (x - x.mean()) / sd if sd > 0 else x * np.nan

def build(d, spec):
    """spec: list of (label, col, transform). transforms: raw, z, iqr, center, cat (dummies, drop first)."""
    cols = {}
    for lab, c, tr in spec:
        x = d[c].astype(float) if tr != "cat" else d[c]
        if tr == "raw": cols[lab] = x
        elif tr == "z": cols[lab] = zs(x)
        elif tr == "iqr":
            iqr = x.quantile(.75) - x.quantile(.25); cols[lab] = x / iqr if iqr > 0 else x * np.nan
        elif tr == "center": cols[lab] = x - x.mean()
        elif tr == "cat":
            lv = sorted(x.unique())
            for v in lv[1:]: cols[f"{lab}_{v}"] = (x == v).astype(float)
        elif callable(tr): cols[lab] = tr(d)
    return pd.DataFrame(cols, index=d.index)

def fit(df, y, spec, focus, need=(), min_n=15, cond=None, prep=None, label=""):
    rows = []
    needcols = [y] + [c for _, c, _ in spec if isinstance(c, str)] + list(need) + list(getattr(prep, "need", []))
    needcols = [c for c in needcols if c in df.columns]
    for pid, d in df.groupby("pid"):
        d = d.dropna(subset=list(dict.fromkeys(needcols)))
        if prep is not None: d = prep(d)
        if d is None or len(d) < min_n: continue
        if cond is not None and not cond(d): continue
        X = build(d, spec)
        if any(f not in X.columns or not np.isfinite(X[f]).all() or X[f].std() < 1e-9 for f in focus): continue
        keep = [c for c in X.columns if X[c].std() > 1e-9]
        X = X[keep]
        Xm = np.column_stack([np.ones(len(d)), X.to_numpy(float)])
        if np.linalg.matrix_rank(Xm) < Xm.shape[1]:
            # drop non-focus columns until full rank
            for c in [c for c in reversed(keep) if c not in focus]:
                X = X.drop(columns=c); Xm = np.column_stack([np.ones(len(d)), X.to_numpy(float)])
                if np.linalg.matrix_rank(Xm) == Xm.shape[1]: break
            if np.linalg.matrix_rank(Xm) < Xm.shape[1]: continue
        yy = d[y].to_numpy(float)
        b, se = ols(yy, Xm, d.workday.to_numpy())
        r = dict(pid=pid, city=d.city.iloc[0], n=len(d), sd_y=yy.std(ddof=1), n_days=d.workday.nunique())
        for f in focus:
            j = 1 + list(X.columns).index(f); r["b_" + f] = b[j]; r["se_" + f] = se[j]
        rows.append(r)
    return pd.DataFrame(rows)

RES = {}       # model id -> dict (pooled results per focus)
PER = {}       # model id -> per-rider frame
NMODELS = [0]
def run(mid, df, y, spec, focus, **kw):
    pr = fit(df, y, spec, focus, **kw)
    PER[mid] = pr
    out = {}
    for f in focus:
        if len(pr) < 3: out[f] = None; continue
        p = pool(pr["b_" + f], pr["se_" + f]); p["n"] = int(pr.n.sum()); out[f] = p
    RES[mid] = out; NMODELS[0] += 1
    return out

def fmt(p, d=2):
    if p is None: return "not estimable"
    return (f"{p['mu']:+.{d}f} [{p['lo_hk']:+.{d}f}, {p['hi_hk']:+.{d}f}], p_HKSJ {p['p_hk']:.3f} "
            f"(DL p {p['p_dl']:.3f}); k {p['k']}, n {p['n']}, I² {p['i2']:.0f}%, β<0 in {p['n_neg']}/{p['k']}")

def shrink(a, b):
    return 1 - abs(b["mu"]) / abs(a["mu"]) if a and b and a["mu"] != 0 else float("nan")

T = [("days30", "days30", "raw"), ("logsess", "logsess", "raw")]
B = T + [("clock", "clock_hour", "raw")]
W = [("hos", "hos", "raw"), ("km10", "km10", "raw"), ("clock", "clock_hour", "raw"),
     ("tat", "tests_already_today", "raw")] + T

OUT = []
def P(s=""): OUT.append(s)

E = EX["sameday"]; EA = EX["after"]
# ===================== F1 =====================
P("## F1 PM2.5 (same-day mean, within-rider z) and vigilance")
for y in ["Y1", "Y2", "Y3"]:
    r = run(f"F1_{y}", D, y, [("pm", E, "z")] + B, ["pm"])
    P(f"- F1 {y} per +1 SD: {fmt(r['pm'])}")
f1 = RES["F1_Y1"]["pm"]; f1y2 = RES["F1_Y2"]["pm"]
med_rt = D.med_rt.median()
def ms(pct): return 1000 / (1000 / med_rt * (1 + pct / 100)) - med_rt  # % speed change -> ms at median RT
P(f"- Bound (Y1, HKSJ 95% limit in harm direction): +1 SD day ≥ {f1['lo_hk']:+.2f}% speed "
  f"(≈ {ms(f1['lo_hk']):+.1f} ms at cohort median RT {med_rt:.0f} ms); +2 SD day ≥ {2*f1['lo_hk']:+.2f}% (≈ {ms(2*f1['lo_hk']):+.1f} ms)")
P(f"- Bound (Y2 lapses): +1 SD ≤ {f1y2['hi_hk']:+.2f} pp; +2 SD ≤ {2*f1y2['hi_hk']:+.2f} pp (cohort mean lapse {D.Y2.mean():.1f}%)")
r = run("F1_Y1_iqr", D, "Y1", [("pm", E, "iqr")] + B, ["pm"]); P(f"- Y1 per within-rider IQR: {fmt(r['pm'])}")
def tert_prep(col):
    def f(d):
        d = d.copy(); q = d[col].rank(method="first") / len(d)
        d["hi"] = (q > 2 / 3).astype(float); d["mid"] = ((q > 1 / 3) & (q <= 2 / 3)).astype(float); return d
    f.need = [col]
    return f
tcond = lambda d: d.hi.sum() >= 3 and (1 - d.hi - d["mid"]).sum() >= 3
r = run("F1_Y1_tert", D, "Y1", [("hi", "hi", "raw"), ("mid", "mid", "raw")] + B, ["hi"], prep=tert_prep(E), cond=tcond)
P(f"- Y1 own top vs bottom tertile: {fmt(r['hi'])}")
r = run("F1_Y1_unadj", D, "Y1", [("pm", E, "z")], ["pm"]); P(f"- Y1 unadjusted: {fmt(r['pm'])}")
r = run("F1_Y1_temp", D, "Y1", [("pm", E, "z"), ("temp", "temp_h", "z")] + B, ["pm"]); P(f"- Y1 + housing temperature (z): {fmt(r['pm'])}")
if EX["cum"]:
    r = run("F1_Y1_cum", D, "Y1", [("pm", EX["cum"], "z")] + B, ["pm"]); P(f"- Y1 cumulative dose (secondary; confounded with time): {fmt(r['pm'])}")

# ===================== EQ =====================
P("\n## EQ Equivalence (TOST), SESOI ±0.2 within-rider SD of outcome per +1 SD PM")
EQ = {}
for y in ["Y1", "Y2", "Y3"]:
    pr = PER[f"F1_{y}"]
    bz = pr.b_pm / pr.sd_y; sz = pr.se_pm / pr.sd_y
    p90 = pool(bz, sz, q=0.95); p95 = pool(bz, sz)
    k = p90["k"]; se = p90["se_hk"]; mu = p90["mu"]
    tl = (mu + 0.2) / se; tu = (0.2 - mu) / se
    def one(t): return 0.5 * p_t(t, k - 1) if t > 0 else 1 - 0.5 * p_t(t, k - 1)
    p_tost = max(one(tl), one(tu)); eqv = p90["lo_hk"] > -0.2 and p90["hi_hk"] < 0.2
    EQ[y] = dict(mu=mu, lo90=p90["lo_hk"], hi90=p90["hi_hk"], p_tost=p_tost, equivalent=bool(eqv), k=k)
    NMODELS[0] += 1
    P(f"- {y}: {mu:+.3f} SD, 90% CI [{p90['lo_hk']:+.3f}, {p90['hi_hk']:+.3f}], TOST p {p_tost:.4f} → {'EQUIVALENT (effects beyond ±0.2 SD ruled out)' if eqv else 'not shown equivalent'}")

# ===================== S exposure sensitivities =====================
P("\n## S Exposure sensitivities (replace same-day mean; set B)")
for nm in ["median", "clip", "onshift", "peak5"]:
    if not EX[nm]: continue
    for y in ["Y1", "Y2", "Y3"]:
        r = run(f"S_{nm}_{y}", D, y, [("pm", EX[nm], "z")] + B, ["pm"])
        P(f"- S-{nm} {y}: {fmt(r['pm'])}")
for y in ["Y1", "Y2", "Y3"]:
    r = run(f"S_cov_{y}", D, y, [("pm", E + "_nocov", "z")] + B, ["pm"]); P(f"- S-cov (no coverage rule) {y}: {fmt(r['pm'])}")
for y in ["Y1", "Y2", "Y3"]:
    r = run(f"S_month_{y}", D, y, [("pm", E, "z")] + B + [("mon", "month", "cat")], ["pm"]); P(f"- S-month (rider×month FE) {y}: {fmt(r['pm'])}")

# ===================== H home vs road =====================
P("\n## H Home vs on-road exposure")
if EX["preshift"] and EX["onshift"]:
    r = run("H1_Y1", D, "Y1", [("pre", EX["preshift"], "z"), ("on", EX["onshift"], "z")] + B, ["pre", "on"])
    P(f"- H1 Y1 home-before-work PM (z): {fmt(r['pre'])}"); P(f"- H1 Y1 on-road PM (z), same model: {fmt(r['on'])}")
for y in ["Y1", "Y3"]:
    if EX["onshift"]:
        r = run(f"H_onshift_T1_{y}", D, y, [("hi", "hi", "raw"), ("mid", "mid", "raw")] + B, ["hi"], prep=tert_prep(EX["onshift"]), cond=tcond)
        P(f"- On-shift PM own top vs bottom tertile, {y}: {fmt(r['hi'])}")

# ===================== F2 fatigue =====================
P("\n## F2 Fatigue gradient (Y1, % of own mean speed)")
r = run("F2a", D, "Y1", [("hos", "hos", "raw")] + T, ["hos", "days30", "logsess"])
P(f"- **F2a hours on shift** (per hour): {fmt(r['hos'], 3)}")
P(f"- F2a T: days/30 {fmt(r['days30'])}; log(session) {fmt(r['logsess'])}")
for lab, c in [("km10", "km10"), ("clock", "clock_hour"), ("tat", "tests_already_today")]:
    r = run(f"F2b_{lab}", D, "Y1", [(lab, c, "raw")] + T, [lab]); P(f"- F2b {lab}: {fmt(r[lab], 3)}")
r = run("F2c", D, "Y1", W, ["hos", "km10", "clock", "tat", "days30"])
for f in ["hos", "km10", "clock", "tat", "days30"]: P(f"- F2c full model {f}: {fmt(r[f], 3)}")
# within-rider VIF
def vif(cols):
    Z = D.dropna(subset=cols + ["Y1"])[["pid"] + cols].copy()
    for c in cols: Z[c] = Z[c] - Z.groupby("pid")[c].transform("mean")
    out = {}
    for c in cols:
        o = [x for x in cols if x != c]; X = np.column_stack([np.ones(len(Z))] + [Z[x] for x in o]); yv = Z[c].to_numpy()
        b = np.linalg.lstsq(X, yv, rcond=None)[0]; r2 = 1 - ((yv - X @ b) ** 2).sum() / ((yv - yv.mean()) ** 2).sum()
        out[c] = 1 / (1 - r2)
    return out
V = vif(["hos", "km10", "clock_hour", "tests_already_today", "days30", "logsess"])
P("- F2c within-rider VIF: " + ", ".join(f"{k} {v:.1f}" for k, v in V.items()))
r = run("F2_sep1", D, "Y1", [("hos", "hos", "raw"), ("clock", "clock_hour", "raw")] + T, ["hos", "clock"])
P(f"- Separation 1 (joint hos + clock): hos {fmt(r['hos'], 3)}; clock {fmt(r['clock'], 3)}")
pre = D[D.test_before_shift == True]
r = run("F2_sep2", pre, "Y1", [("clock", "clock_hour", "raw")] + T, ["clock"])
P(f"- Separation 2 (pre-shift tests only, clock-hour slope without work): {fmt(r['clock'], 3)}")
on = D[(D.test_before_shift == False) & (D.hos > 0)]
r = run("F2_sep3", on, "Y1", [("hos", "hos", "raw"), ("clock", "clock_hour", "raw")] + T, ["hos", "clock"])
P(f"- Separation 3 (on-shift tests, hos at fixed clock hour): hos {fmt(r['hos'], 3)}; clock {fmt(r['clock'], 3)}")
r = run("F2_onshift_clock", on, "Y1", [("clock", "clock_hour", "raw")] + T, ["clock"])
P(f"- (comparison) on-shift tests, clock-hour slope alone: {fmt(r['clock'], 3)}")
# within-day pairs
pr_rows = []
for (pid, wd), g in D.dropna(subset=["Y1", "hos"]).groupby(["pid", "workday"]):
    g = g.sort_values("ts")
    for i in range(1, len(g)):
        a, b_ = g.iloc[i - 1], g.iloc[i]
        pr_rows.append(dict(pid=pid, workday=wd, city=a.city, dY=b_.Y1 - a.Y1, dH=b_.hos - a.hos, dC=b_.clock_hour - a.clock_hour))
PAIRS = pd.DataFrame(pr_rows)
r = run("F2d", PAIRS, "dY", [("dH", "dH", "raw")], ["dH"])
P(f"- **F2d within rider-day** ΔY1 per Δhour on shift ({len(PAIRS)} consecutive pairs): {fmt(r['dH'], 3)}")
P(f"- F2d: share of pairs with ΔH = ΔClock (± 0.05 h): {(abs(PAIRS.dH - PAIRS.dC) < 0.05).mean():.0%}")
r = run("F2a_onshiftonly", on, "Y1", [("hos", "hos", "raw")] + T, ["hos"]); P(f"- F2a sens., pre-shift tests excluded: {fmt(r['hos'], 3)}")
r = run("F2a_daysonly", D, "Y1", [("hos", "hos", "raw"), ("days30", "days30", "raw")], ["hos"]); P(f"- F2a sens., days only (no log session): {fmt(r['hos'], 3)}")
r = run("F2a_sensoron", D, "Y1", [("son", "sensor_on_hours_before_test", "raw")] + T, ["son"]); P(f"- F2a sens., sensor-on hours instead of GPS hours: {fmt(r['son'], 3)}")
# fatigue curve data (figure 1)
bins = [-0.01, 0.001, 2, 4, 6, 8, 10, 99]; blab = ["pre-shift", "0–2", "2–4", "4–6", "6–8", "8–10", "≥10"]
D["hbin"] = pd.cut(D.hos, bins, labels=blab)
D.loc[D.test_before_shift == True, "hbin"] = "pre-shift"
rb = D.dropna(subset=["hbin", "Y1"]).groupby(["pid", "hbin"], observed=True).Y1.mean().reset_index()
curve = rb.groupby("hbin", observed=True).Y1.agg(["mean", "std", "count"]).reindex(blab)
curve["lo"] = curve["mean"] - 1.96 * curve["std"] / np.sqrt(curve["count"]); curve["hi"] = curve["mean"] + 1.96 * curve["std"] / np.sqrt(curve["count"])
P("- Fatigue curve (mean of rider-bin means of Y1, 95% CI across riders):")
for b_, rr in curve.iterrows(): P(f"  - {b_}: {rr['mean']:+.2f}% [{rr.lo:+.2f}, {rr.hi:+.2f}], riders {int(rr['count'])}")

# ===================== F3 placebo =====================
P("\n## F3 Placebo (PM after the test, z)")
needW = ["hos", "km10", "clock_hour", "tests_already_today"]
F3 = {}
for y in ["Y1", "Y2", "Y3"]:
    a = run(f"F3a_{y}", D, y, [("pa", EA, "z")] + B, ["pa"], need=needW)["pa"]
    b_ = run(f"F3b_{y}", D, y, [("pa", EA, "z")] + B + W[:4], ["pa"], need=needW)["pa"]
    F3[y] = (a, b_)
    P(f"- F3a {y} placebo alone: {fmt(a)}"); P(f"- F3b {y} placebo + workday: {fmt(b_)}"); P(f"- {y} shrinkage = {shrink(a, b_):.0%}")
c_ = run("F3c", D, "Y1", [("pm", E, "z")] + B, ["pm"], need=needW)["pm"]
d_ = run("F3d", D, "Y1", [("pm", E, "z")] + B + W[:4], ["pm"], need=needW)["pm"]
P(f"- F3c same-day PM (same sessions): {fmt(c_)}"); P(f"- F3d same-day PM + workday: {fmt(d_)}; shrinkage {shrink(c_, d_):.0%}")
e_ = run("F3e", D, "Y1", [("pm", E, "z"), ("pa", EA, "z")] + B, ["pm", "pa"])
P(f"- F3e joint: same-day {fmt(e_['pm'])}; after {fmt(e_['pa'])}")
# H2 split placebos
for nm in ["after_onshift", "after_home"]:
    if EX[nm]:
        a = run(f"H2a_{nm}", D, "Y1", [("pa", EX[nm], "z")] + B, ["pa"], need=needW)["pa"]
        b_ = run(f"H2b_{nm}", D, "Y1", [("pa", EX[nm], "z")] + B + W[:4], ["pa"], need=needW)["pa"]
        P(f"- H2 {nm} alone: {fmt(a)}"); P(f"- H2 {nm} + workday: {fmt(b_)}; shrinkage {shrink(a, b_):.0%}")
r = run("F3a_month", D, "Y1", [("pa", EA, "z")] + B + [("mon", "month", "cat")], ["pa"], need=needW)["pa"]
r2 = run("F3b_month", D, "Y1", [("pa", EA, "z")] + B + W[:4] + [("mon", "month", "cat")], ["pa"], need=needW)["pa"]
P(f"- S-month F3a: {fmt(r)}; F3b: {fmt(r2)}")

# ===================== F4 =====================
P("\n## F4 Does PM steepen the fatigue gradient? (hours_on_shift centred × PM z)")
def ixprep(d):
    d = d.copy(); d["hc"] = d.hos - d.hos.mean(); d["pz"] = zs(d[E]); d["ix"] = d.hc * d.pz; return d
r = run("F4", D, "Y1", [("hc", "hc", "raw"), ("pz", "pz", "raw"), ("ix", "ix", "raw")] + T, ["ix", "hc", "pz"], need=["hos", E], prep=ixprep)
P(f"- F4 interaction (% per hour per SD): {fmt(r['ix'], 3)}"); P(f"- F4 hours main effect: {fmt(r['hc'], 3)}; PM main: {fmt(r['pz'])}")

# ===================== M1 moving =====================
P("\n## M1 Tests taken while the sensor was moving (P8)")
ev = D[D.P8_moving.notna()].copy(); ev["mv"] = ev.P8_moving.astype(float)
def wilson(k, n):
    if n == 0: return (np.nan, np.nan)
    p = k / n; z = 1.96; den = 1 + z * z / n; c = (p + z * z / (2 * n)) / den; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h
k_, n_ = int(ev.mv.sum()), len(ev); lo, hi = wilson(k_, n_)
P(f"- Cohort: {k_}/{n_} GPS-evaluable tests = {k_/n_:.1%} [{lo:.1%}, {hi:.1%}]; riders with ≥1: {(ev.groupby('pid').mv.sum() > 0).sum()}/22")
for cty, g in ev.groupby("city"):
    k2, n2 = int(g.mv.sum()), len(g); l2, h2 = wilson(k2, n2); P(f"  - {cty}: {k2}/{n2} = {k2/n2:.1%} [{l2:.1%}, {h2:.1%}]")
byr = ev.groupby("pid").mv.agg(["sum", "count"]); byr["share"] = byr["sum"] / byr["count"]
P("- By rider (moving/evaluable): " + "; ".join(f"{p} {int(r_['sum'])}/{int(r_['count'])}" for p, r_ in byr.iterrows()))
P("- By clock band: " + "; ".join(f"{b_} {g.mv.mean():.0%} (n {len(g)})" for b_, g in ev.groupby("band")))
ev["hbin"] = D.loc[ev.index, "hbin"]
P("- By hours-on-shift bin: " + "; ".join(f"{b_} {g.mv.mean():.0%} (n {len(g)})" for b_, g in ev.groupby("hbin", observed=True)))
P(f"- Median sensor speed in P8 tests: see QC_SUMMARY (23.5 km/h)")
mcond = lambda d: d.mv.sum() >= 3 and (1 - d.mv).sum() >= 3
for y in ["Y1", "Y2", "Y3"]:
    r = run(f"M1_{y}", ev, y, [("mv", "mv", "raw")] + T + [("clock", "clock_hour", "raw")], ["mv"], cond=mcond)
    P(f"- M1 {y} moving − not moving (within rider): {fmt(r['mv'])}")
BYR_MOVE = byr

# ===================== M2 timing =====================
P("\n## M2 Timing (clock-hour bands)")
bands = ["04-10", "10-14", "14-18", "18-22", "22-04"]
share = pd.crosstab(D.pid, D.band, normalize="index").reindex(columns=bands, fill_value=0)
P("- Cohort share of tests: " + "; ".join(f"{b_} {(D.band == b_).mean():.0%}" for b_ in bands))
for cty, g in D.groupby("city"): P(f"  - {cty}: " + "; ".join(f"{b_} {(g.band == b_).mean():.0%}" for b_ in bands))
P("- Riders with ≥ 20% of tests 22–04: " + str(int((share["22-04"] >= .2).sum())))
# residualise Y1 on T within rider
res = []
for pid, d in D.dropna(subset=["Y1"]).groupby("pid"):
    X = np.column_stack([np.ones(len(d)), d.days30, d.logsess]); b = np.linalg.lstsq(X, d.Y1.to_numpy(), rcond=None)[0]
    dd = d[["pid", "band"]].copy(); dd["r"] = d.Y1.to_numpy() - X @ b; res.append(dd)
RS = pd.concat(res)
M2 = {}
for b_ in bands:
    g = RS[RS.band == b_].groupby("pid").r.agg(["mean", "std", "count"]); g = g[g["count"] >= 3]
    p = pool(g["mean"], g["std"] / np.sqrt(g["count"])) if len(g) >= 3 else None
    if p: p["n"] = int(g["count"].sum())
    M2[b_] = p; NMODELS[0] += 1
    P(f"- M2 Y1 in band {b_} vs own mean: {fmt(p)}")
SHARE = share

# ===================== C1 city =====================
P("\n## C1 City")
C1 = {}
for mid, f in [("F1_Y1", "pm"), ("F1_Y2", "pm"), ("F1_Y3", "pm"), ("F2a", "hos")]:
    pr = PER[mid]; ps = {}
    for cty in ["BKK", "CMI"]:
        g = pr[pr.city == cty]; ps[cty] = pool(g["b_" + f], g["se_" + f]); NMODELS[0] += 1
    diff = ps["CMI"]["mu"] - ps["BKK"]["mu"]; se = math.sqrt(ps["CMI"]["se_dl"] ** 2 + ps["BKK"]["se_dl"] ** 2)
    k = ps["CMI"]["k"] + ps["BKK"]["k"]; pdiff = p_t(diff / se, k - 2)
    C1[mid] = (ps, diff, se, pdiff)
    for cty in ["BKK", "CMI"]:
        ps[cty]["n"] = int(pr[pr.city == cty].n.sum()); P(f"- {mid} {cty}: {fmt(ps[cty], 3)}")
    tc = t_crit(k - 2); P(f"- {mid} CMI − BKK: {diff:+.3f} [{diff - tc*se:+.3f}, {diff + tc*se:+.3f}], p {pdiff:.3f}")

# ===================== T1 slow tail =====================
P("\n## T1 Slow tail: own top vs bottom tertile of same-day PM, with PM-after placebo")
T1 = {}
for y in ["Y3", "Y2"]:
    a = run(f"T1_{y}", D, y, [("hi", "hi", "raw"), ("mid", "mid", "raw")] + B, ["hi"], prep=tert_prep(E), cond=tcond)["hi"]
    b_ = run(f"T1p_{y}", D, y, [("hi", "hi", "raw"), ("mid", "mid", "raw")] + B, ["hi"], prep=tert_prep(EA), cond=tcond)["hi"]
    T1[y] = (a, b_)
    P(f"- T1 {y} high vs low same-day PM: {fmt(a)}"); P(f"- T1 placebo {y} high vs low PM-after: {fmt(b_)}")
# SPEC §3 S applied to T1-Y3 (added 2026-11-10 after S7 audit finding S3-4; these models were pre-specified but omitted)
for nm, col in [("median", EX["median"]), ("clip", EX["clip"]), ("peak5", EX["peak5"]), ("nocov", E + "_nocov")]:
    if not col: continue
    r = run(f"S_{nm}_T1_Y3", D, "Y3", [("hi", "hi", "raw"), ("mid", "mid", "raw")] + B, ["hi"], prep=tert_prep(col), cond=tcond)["hi"]
    P(f"- S-{nm} T1-Y3 high vs low: {fmt(r)}")

# ===================== PH phone/step changes =====================
P("\n## PH Possible RT step changes (R02 from 2026-01-21, R08 from 2026-03-11, R22 from 2026-03-06)")
STEP = {"R02": "2026-01-21", "R08": "2026-03-11", "R22": "2026-03-06"}
D["period"] = 0.0
for p_, dt in STEP.items(): D.loc[(D.pid == p_) & (D.ts >= pd.Timestamp(dt)), "period"] = 1.0
PERI = [("period", "period", "raw")]
r = run("PH_F1", D, "Y1", [("pm", E, "z")] + B + PERI, ["pm"]); P(f"- PH F1-Y1: {fmt(r['pm'])}")
r = run("PH_F2a", D, "Y1", [("hos", "hos", "raw")] + T + PERI, ["hos"]); P(f"- PH F2a: {fmt(r['hos'], 3)}")
a = run("PH_F3a", D, "Y1", [("pa", EA, "z")] + B + PERI, ["pa"], need=needW)["pa"]
b_ = run("PH_F3b", D, "Y1", [("pa", EA, "z")] + B + W[:4] + PERI, ["pa"], need=needW)["pa"]
P(f"- PH F3a: {fmt(a)}; F3b: {fmt(b_)}; shrinkage {shrink(a, b_):.0%}")
Dn = D[~D.pid.isin(STEP)]
r = run("PHd_F1", Dn, "Y1", [("pm", E, "z")] + B, ["pm"]); P(f"- PH-drop F1-Y1: {fmt(r['pm'])}")
r = run("PHd_F2a", Dn, "Y1", [("hos", "hos", "raw")] + T, ["hos"]); P(f"- PH-drop F2a: {fmt(r['hos'], 3)}")
a = run("PHd_F3a", Dn, "Y1", [("pa", EA, "z")] + B, ["pa"], need=needW)["pa"]
b_ = run("PHd_F3b", Dn, "Y1", [("pa", EA, "z")] + B + W[:4], ["pa"], need=needW)["pa"]
P(f"- PH-drop F3a: {fmt(a)}; F3b: {fmt(b_)}; shrinkage {shrink(a, b_):.0%}")

# ===================== O1 sensitivity =====================
P("\n## O1 Sensitivity")
def targets(df, tag):
    o = {}
    o["F1"] = run(f"O1{tag}_F1", df, "Y1", [("pm", E, "z")] + B, ["pm"])["pm"]
    o["F2a"] = run(f"O1{tag}_F2a", df, "Y1", [("hos", "hos", "raw")] + T, ["hos"])["hos"]
    a = run(f"O1{tag}_F3a", df, "Y1", [("pa", EA, "z")] + B, ["pa"], need=needW)["pa"]
    b_ = run(f"O1{tag}_F3b", df, "Y1", [("pa", EA, "z")] + B + W[:4], ["pa"], need=needW)["pa"]
    o["F3a"] = a; o["F3b"] = b_; o["shrink"] = shrink(a, b_)
    o["T1"] = run(f"O1{tag}_T1", df, "Y3", [("hi", "hi", "raw"), ("mid", "mid", "raw")] + B, ["hi"], prep=tert_prep(E), cond=tcond)["hi"]
    o["T1p"] = run(f"O1{tag}_T1p", df, "Y3", [("hi", "hi", "raw"), ("mid", "mid", "raw")] + B, ["hi"], prep=tert_prep(EA), cond=tcond)["hi"]
    return o
O1 = {}
for tag, df in [("a_noexcl", D[D.any_exclusion != True]), ("b_noP8", D[D.P8_moving != True]), ("c_noR12R14", D[D.late_starter_handover != True])]:
    o = targets(df, tag); O1[tag] = o
    P(f"- ({tag}) n {len(df)}: F1-Y1 {fmt(o['F1'])}")
    P(f"  - F2a {fmt(o['F2a'], 3)}"); P(f"  - F3a {fmt(o['F3a'])}; F3b {fmt(o['F3b'])}; shrinkage {o['shrink']:.0%}")
    P(f"  - T1-Y3 {fmt(o['T1'])}; T1 placebo {fmt(o['T1p'])}")
LOO = []
nm0 = NMODELS[0]
for p_ in sorted(D.pid.unique()):
    o = targets(D[D.pid != p_], "_loo"); LOO.append(dict(pid=p_, F1=o["F1"]["mu"], F1_lo=o["F1"]["lo_hk"], F1_hi=o["F1"]["hi_hk"],
        F2a=o["F2a"]["mu"], F2a_lo=o["F2a"]["lo_hk"], F2a_hi=o["F2a"]["hi_hk"], F2a_p=o["F2a"]["p_hk"],
        F3a=o["F3a"]["mu"], F3a_p=o["F3a"]["p_hk"], shrink=o["shrink"], T1=o["T1"]["mu"], T1_p=o["T1"]["p_hk"], T1p_p=o["T1p"]["p_hk"]))
NMODELS[0] = nm0 + 6  # LOO counted once per target
LOO = pd.DataFrame(LOO)
for c in ["F1", "F2a", "F3a", "shrink", "T1"]:
    P(f"- (d) LOO {c}: range {LOO[c].min():+.3f} to {LOO[c].max():+.3f}")
P(f"- (d) LOO F2a p_HKSJ max {LOO.F2a_p.max():.3f}; F2a CI excludes 0 in {(LOO.F2a_hi < 0).sum() + (LOO.F2a_lo > 0).sum()}/22 refits")
P(f"- (d) LOO F3a p_HKSJ range {LOO.F3a_p.min():.3f}–{LOO.F3a_p.max():.3f}; T1 p range {LOO.T1_p.min():.3f}–{LOO.T1_p.max():.3f}; T1 placebo p range {LOO.T1p_p.min():.3f}–{LOO.T1p_p.max():.3f}")

# ===================== B Bangkok pilot =====================
P("\n## B Bangkok demographic pilot (n = 10; validity and description only)")
dem = pd.read_csv(os.path.join(ROOT, "demographics", "bkk_demographics.csv"))
rk = D.groupby("pid").agg(speed=("y1_speed_rtgt0", "mean"), medrt=("med_rt", "median")).reset_index().merge(dem[["pid", "sex", "age_2026"]], on="pid")
rng = np.random.default_rng(RNG_SEED)
def spearman(x, y):
    rx = pd.Series(x).rank().to_numpy(); ry = pd.Series(y).rank().to_numpy(); return np.corrcoef(rx, ry)[0, 1]
def slope10(x, y): return np.polyfit(x / 10, y, 1)[0]
B1 = {}
for v in ["speed", "medrt"]:
    x = rk.age_2026.to_numpy(float); y = rk[v].to_numpy(float); n = len(x)
    rho = spearman(x, y); sl = slope10(x, y); bs_r, bs_s = [], []
    for _ in range(10000):
        i = rng.integers(0, n, n)
        if np.unique(x[i]).size < 3: continue
        bs_r.append(spearman(x[i], y[i])); bs_s.append(slope10(x[i], y[i]))
    B1[v] = dict(rho=rho, rho_ci=np.nanpercentile(bs_r, [2.5, 97.5]), slope=sl, slope_ci=np.percentile(bs_s, [2.5, 97.5]))
    m = rk[rk.sex == "M"][v].to_numpy(); f = rk[rk.sex == "F"][v].to_numpy(); bd = []
    for _ in range(10000): bd.append(rng.choice(m, len(m)).mean() - rng.choice(f, len(f)).mean())
    B1[v]["mf"] = m.mean() - f.mean(); B1[v]["mf_ci"] = np.percentile(bd, [2.5, 97.5])
    u = "Hz" if v == "speed" else "ms"
    P(f"- B1 {v} vs age: Spearman ρ {rho:+.2f} [{B1[v]['rho_ci'][0]:+.2f}, {B1[v]['rho_ci'][1]:+.2f}]; slope per 10 y {sl:+.3f} {u} [{B1[v]['slope_ci'][0]:+.3f}, {B1[v]['slope_ci'][1]:+.3f}]")
    P(f"- B1 {v} men − women: {B1[v]['mf']:+.3f} {u} [{B1[v]['mf_ci'][0]:+.3f}, {B1[v]['mf_ci'][1]:+.3f}]")
NMODELS[0] += 4
bands_age = pd.cut(rk.age_2026, [19, 29, 39, 49, 59], labels=["20–29", "30–39", "40–49", "50–59"]).value_counts().sort_index()
P("- Age bands (count): " + ", ".join(f"{k} {v}" for k, v in bands_age.items()) + f"; sex M {int((rk.sex=='M').sum())}, F {int((rk.sex=='F').sum())}")
def metareg(pr, f, covs):
    m = pr.merge(dem[["pid", "sex", "age_2026"]], on="pid")
    b = m["b_" + f].to_numpy(); s = m["se_" + f].to_numpy(); k = len(b)
    tau2 = pool(b, s)["tau2"]; w = 1 / (s ** 2 + tau2)
    cols = [np.ones(k)]
    if "age" in covs: cols.append(m.age_2026.to_numpy() / 10)
    if "sex" in covs: cols.append((m.sex == "M").astype(float).to_numpy())
    X = np.column_stack(cols); Wm = np.diag(w); XtWXi = np.linalg.inv(X.T @ Wm @ X); beta = XtWXi @ X.T @ Wm @ b
    resid = b - X @ beta; p = X.shape[1]; s2 = max((w * resid ** 2).sum() / (k - p), 1.0)
    se = np.sqrt(np.diag(XtWXi) * s2); tc = t_crit(k - p)
    return [(beta[j], beta[j] - tc * se[j], beta[j] + tc * se[j], p_t(beta[j] / se[j], k - p)) for j in range(1, p)], k
BMR = {}
for mid, f, nm in [("F1_Y1", "pm", "B3 PM(z) slope"), ("F2a", "hos", "B4 hours-on-shift slope")]:
    prb = PER[mid][PER[mid].pid.isin(dem.pid)]
    pb = pool(prb["b_" + f], prb["se_" + f]); pb["n"] = int(prb.n.sum()); P(f"- {nm}, within-Bangkok pooled: {fmt(pb, 3)}")
    for covs in [["age"], ["sex"], ["age", "sex"]]:
        res_, k = metareg(prb, f, covs); NMODELS[0] += 1
        P(f"- {nm} on {'+'.join(covs)} (k {k}): " + "; ".join(f"{c} {e[0]:+.3f} [{e[1]:+.3f}, {e[2]:+.3f}] p {e[3]:.3f}" for c, e in zip(covs if len(covs) > 1 else covs, res_)))
        BMR[(nm, "+".join(covs))] = res_

# ===================== 20-slot sensitivity (added 2026-11-11, user decision "22 primary, 20 sensitivity", after results) =====================
P("\n## 20-slot sensitivity (without R12, R14; added after results, user decision Nov 11)")
D20 = D[D.late_starter_handover != True]
for y in ["Y1", "Y2", "Y3"]:
    r = run(f"S20_F1_{y}", D20, y, [("pm", E, "z")] + B, ["pm"])["pm"]
    pr = PER[f"S20_F1_{y}"]; p90 = pool(pr.b_pm / pr.sd_y, pr.se_pm / pr.sd_y, q=0.95); NMODELS[0] += 1
    se = p90["se_hk"]; mu = p90["mu"]; k = p90["k"]
    def one(t): return 0.5 * p_t(t, k - 1) if t > 0 else 1 - 0.5 * p_t(t, k - 1)
    pt = max(one((mu + 0.2) / se), one((0.2 - mu) / se)); eqv = p90["lo_hk"] > -0.2 and p90["hi_hk"] < 0.2
    P(f"- 20-slot F1 {y}: {fmt(r)}")
    P(f"- 20-slot EQ {y}: {mu:+.3f} SD, 90% CI [{p90['lo_hk']:+.3f}, {p90['hi_hk']:+.3f}], TOST p {pt:.4f} → {'EQUIVALENT' if eqv else 'not shown equivalent'}")
    if y == "Y1": P(f"- 20-slot bound (Y1): +1 SD ≥ {r['lo_hk']:+.2f}% (≈ {ms(r['lo_hk']):+.1f} ms); +2 SD ≥ {2*r['lo_hk']:+.2f}%")
ev20 = ev[ev.late_starter_handover != True]
k_, n_ = int(ev20.mv.sum()), len(ev20); lo, hi = wilson(k_, n_)
P(f"- 20-slot M1 prevalence: {k_}/{n_} = {k_/n_:.1%} [{lo:.1%}, {hi:.1%}]; riders with ≥1: {(ev20.groupby('pid').mv.sum() > 0).sum()}/{ev20.pid.nunique()}")
for cty, g in ev20.groupby("city"):
    k2, n2 = int(g.mv.sum()), len(g); l2, h2 = wilson(k2, n2); P(f"  - {cty}: {k2}/{n2} = {k2/n2:.1%} [{l2:.1%}, {h2:.1%}]")
for y in ["Y1", "Y2", "Y3"]:
    r = run(f"S20_M1_{y}", ev20, y, [("mv", "mv", "raw")] + T + [("clock", "clock_hour", "raw")], ["mv"], cond=mcond)["mv"]
    P(f"- 20-slot M1 {y} moving − not moving: {fmt(r)}")

# ===================== Gate 2 rules =====================
P("\n## Gate 2 decision rules (plan §5, SPEC §7)")
f2 = RES["F2a"]["hos"]; a3, b3 = F3["Y1"]; sh = shrink(a3, b3)
r1_ci = f2["hi_hk"] < 0 or f2["lo_hk"] > 0
r1 = (f2["hi_hk"] < 0) and sh >= 0.5
P(f"- Rule 1: F2a {f2['mu']:+.3f}%/h [{f2['lo_hk']:+.3f}, {f2['hi_hk']:+.3f}] (CI excludes 0: {r1_ci}, expected sign −: {f2['mu'] < 0}); "
  f"placebo shrinkage {sh:.0%} (F3a {a3['mu']:+.3f}, p {a3['p_hk']:.3f} → F3b {b3['mu']:+.3f}, p {b3['p_hk']:.3f}) → **{'HOLDS' if r1 else 'does not hold'}**")
t1, t1p = T1["Y3"]
r2 = (t1["mu"] < 0) and (t1p["lo_hk"] < 0 < t1p["hi_hk"])
P(f"- Rule 2: T1-Y3 {t1['mu']:+.2f}% [{t1['lo_hk']:+.2f}, {t1['hi_hk']:+.2f}] (harm direction: {t1['mu'] < 0}; own CI excludes 0: {t1['hi_hk'] < 0}); "
  f"placebo {t1p['mu']:+.2f}% [{t1p['lo_hk']:+.2f}, {t1p['hi_hk']:+.2f}] (null: {t1p['lo_hk'] < 0 < t1p['hi_hk']}) → **{'HOLDS as written' if r2 else 'does not hold'}**")
P(f"- Rule 3 (claim c): {'applies' if not (r1 or r2) else 'not needed'}")
P(f"\nModels run this script (Stage-2 pooled estimates; LOO counted once per target): {NMODELS[0]}")
if LOG: P("\nLog: " + "; ".join(sorted(set(LOG))))

# ===================== save =====================
with open(os.path.join(A.out, "results_tables.md"), "w") as fh:
    fh.write(f"# S3 run output (--pm-col {PM}); auto-generated by scripts/pvt_analysis.py\n\n" + "\n".join(OUT) + "\n")
allper = []
for mid, pr in PER.items():
    if len(pr) and not mid.startswith("O1_loo"):
        x = pr.copy(); x.insert(0, "model", mid); allper.append(x)
pd.concat(allper).to_csv(os.path.join(A.out, "per_rider_estimates.csv"), index=False)
LOO.to_csv(os.path.join(A.out, "loo.csv"), index=False)
curve.to_csv(os.path.join(A.out, "fatigue_curve.csv"))
rb.to_csv(os.path.join(A.out, "fatigue_rider_bins.csv"), index=False)
BYR_MOVE.to_csv(os.path.join(A.out, "moving_by_rider.csv"))
SHARE.to_csv(os.path.join(A.out, "band_share_by_rider.csv"))
pooled = {mid: {f: v for f, v in d.items()} for mid, d in RES.items() if not mid.startswith("O1_loo")}
json.dump(dict(pooled=pooled, EQ=EQ, M2=M2, nmodels=NMODELS[0], pm_col=PM,
               cal=dict(pm_sameday_median=float(D[E].median()), pm_sameday_rider_sd_median=float(D.groupby("pid")[E].std().median()),
                        pm_sameday_rider_iqr_median=float(D.groupby("pid")[E].quantile(.75).sub(D.groupby("pid")[E].quantile(.25)).median()),
                        pm_after_median=float(D[EA].median()), pm_after_rider_sd_median=float(D.groupby("pid")[EA].std().median()))),
          open(os.path.join(A.out, "pooled.json"), "w"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
print("\n".join(OUT))
