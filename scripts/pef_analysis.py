"""Paper 2 peak-flow analysis (S4), as specified in peakflow_paper/SPEC_ADDENDUM.md (approved 2026-11-11).

Run once:  python3 -I scripts/pef_analysis.py --pm-col pm
Writes peakflow_paper/<out>/ (per-person CSVs, pooled.json, descriptives.json) and <out>/RESULTS_tables.md.
numpy/pandas only. Person IDs are R##/O## study IDs. PM is raw unless --pm-col points at a calibrated column.
"""
import argparse, json, math, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pef_stats import ols, pool, p_t, t_crit  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEED = 20261111
NBOOT = 10000
SESOI_PP = 2.0      # user-approved, Nov 11
SESOI_SD = 0.2      # secondary row (Paper 1)
MIN_N = 12


# ---------------------------------------------------------------- helpers
def cdf_t(t, df):
    """P(T <= t) for Student t."""
    p2 = p_t(t, df)
    return 1 - p2 / 2 if t > 0 else p2 / 2


def scale(x, how):
    x = x.astype(float)
    if how == "z":
        s = x.std(ddof=1)
        return (x - x.mean()) / s if s > 1e-9 else x * np.nan
    if how == "iqr":
        q = x.quantile(0.75) - x.quantile(0.25)
        return x / q if q > 1e-9 else x * np.nan
    if how == "raw10":
        return x / 10.0
    if how == "none":
        return x
    raise ValueError(how)


def fit_person(dd, y, xs, ctrls, min_n=MIN_N, dummies=None):
    """xs: list of (col, scaling). Returns dict with beta/se for each x term, or None."""
    need = [y] + [c for c, _ in xs] + list(ctrls)
    dd = dd.dropna(subset=need)
    if len(dd) < min_n:
        return None
    cols = [np.ones(len(dd))]
    names = []
    for c, how in xs:
        v = scale(dd[c], how)
        if v.isna().any() or v.std() < 1e-9:
            return None
        cols.append(v.to_numpy(float)); names.append(c)
    for c in ctrls:
        cols.append(dd[c].to_numpy(float))
    if dummies is not None:
        lev = sorted(dd[dummies].unique())
        for L in lev[1:]:
            cols.append((dd[dummies] == L).to_numpy(float))
    X = np.column_stack(cols)
    if np.linalg.matrix_rank(X) < X.shape[1] or len(dd) <= X.shape[1]:
        return None
    yy = dd[y].to_numpy(float)
    b, se, _ = ols(yy, X)
    out = dict(n=len(dd), sd_y=yy.std(ddof=1))
    for i, nm in enumerate(names):
        out[f"b{i}"] = b[1 + i]; out[f"se{i}"] = se[1 + i]
    return out


def pool_rows(t, i=0):
    t = t.dropna(subset=[f"b{i}"])
    if len(t) < 2:
        return None
    r = pool(t[f"b{i}"], t[f"se{i}"])
    r["n"] = int(t.n.sum())
    return {k: (float(v) if isinstance(v, (float, np.floating, int, np.integer)) else v) for k, v in r.items()}


# ---------------------------------------------------------------- data
def load(pm):
    d = pd.read_csv(os.path.join(ROOT, "foundation/linked_v4_pef.csv"),
                    parse_dates=["timestamp_local", "shift_start_local", "last_move_local"])
    d = d[d.role == "rider"].copy()
    n0 = len(d)
    for c in ["exact_duplicate", "F1_low", "F5_outside_enrolment", "F3_poor_reproducibility", "F4_right_after_pvt"]:
        d[c] = d[c].astype(str).str.lower().eq("true")
    d = d[~d.exact_duplicate & ~d.F1_low & ~d.F5_outside_enrolment]
    n1 = len(d)
    d = d[d.period.isin(["morning", "evening"])].copy()
    h = d.timestamp_local.dt.hour + d.timestamp_local.dt.minute / 60
    d["clock_u"] = np.where(h < 4, h + 24, h)
    d["month"] = pd.to_datetime(d.workday).dt.strftime("%Y-%m")
    coh = pd.read_csv(os.path.join(ROOT, "foundation/cohort.csv"))
    coh = coh[coh.role == "rider"][["pid", "city", "slot_original"]].rename(columns={"city": "city_c"})
    coh["slot20"] = coh.slot_original.astype(str).str.lower().eq("true")
    d = d.merge(coh[["pid", "city_c", "slot20"]], on="pid", how="left")
    # PM columns for the chosen --pm-col
    def col(suf):
        c = f"{pm}_{suf}"
        return d[c] if c in d.columns else pd.Series(np.nan, index=d.index)
    d["PM_sd"] = col("sameday").where(d.cov_ok_sameday.astype(str).str.lower().eq("true"))
    d["PM_sd_nocov"] = col("sameday")
    d["PM_after"] = col("after").where(d.cov_ok_after.astype(str).str.lower().eq("true"))
    d["PM_cum"] = col("sameday_cum").where(d.cov_ok_sameday.astype(str).str.lower().eq("true"))
    d["PM_med"] = col("sameday_median").where(d.cov_ok_sameday.astype(str).str.lower().eq("true"))
    d["PM_clip"] = col("sameday_clip").where(d.cov_ok_sameday.astype(str).str.lower().eq("true"))
    d["PM_on"] = col("onshift").where(d.cov_ok_onshift.astype(str).str.lower().eq("true"))
    d["PM_sd_any"] = col("sameday")  # for between-reading means (morning windows can be short)
    d["km10"] = d.km_ridden_before_test / 10.0
    d["F4"] = d.F4_right_after_pvt.astype(float)
    return d, dict(rider_rows=n0, after_exclusions=n1, morning_evening=len(d))


def select(d, rule):
    if rule == "highest":
        s = d.sort_values(["peak_flow", "timestamp_local"], ascending=[False, False])
        s = s.drop_duplicates(["pid", "workday", "period"])
    else:
        s = d.sort_values("timestamp_local").drop_duplicates(["pid", "workday", "period"], keep="last")
    pb = s[s.peak_flow >= 50].groupby("pid").peak_flow.apply(lambda v: float(np.percentile(v, 95)))
    s = s.copy()
    s["PB"] = s.pid.map(pb)
    s["pct"] = s.peak_flow / s.PB * 100
    return s.sort_values(["pid", "timestamp_local"]), pb


# ---------------------------------------------------------------- models
class Runner:
    def __init__(self):
        self.models = {}; self.persons = []

    def run(self, name, data, y, xs, ctrls, dummies=None, sets=("all", "slot20"), cities=False, label=None):
        rows = []
        for pid, g in data.groupby("pid"):
            r = fit_person(g, y, xs, ctrls, dummies=dummies)
            if r is None:
                continue
            r.update(pid=pid, city=g.city_c.iloc[0], slot20=bool(g.slot20.iloc[0]))
            rows.append(r)
        t = pd.DataFrame(rows)
        res = dict(label=label or name, terms=[c for c, _ in xs], y=y, ctrls=list(ctrls))
        for i, (c, how) in enumerate(xs):
            for s in sets:
                tt = t if s == "all" else t[t.slot20]
                res[f"{s}:{c}:{how}"] = pool_rows(tt, i) if len(tt) else None
            if cities:
                for cc in ["BKK", "CMI"]:
                    res[f"{cc}:{c}:{how}"] = pool_rows(t[t.city == cc], i) if len(t) else None
        self.models[name] = res
        t2 = t.copy(); t2.insert(0, "model", name); self.persons.append(t2)
        return t, res


def tost(r, m):
    """TOST on the HKSJ scale (k-1 df); equivalence if the HKSJ 90% CI lies inside +-m."""
    df = r["k"] - 1; se = r["se_hk"]; mu = r["mu"]
    p_low = 1 - cdf_t((mu + m) / se, df)   # H0: mu <= -m
    p_up = cdf_t((mu - m) / se, df)        # H0: mu >= +m
    return dict(margin=m, lo90=r["lo90_hk"], hi90=r["hi90_hk"], p_tost=max(p_low, p_up),
                equivalent=bool(r["lo90_hk"] > -m and r["hi90_hk"] < m))


def boot_share_tertiles(ev, persons, rng):
    """Per-person share < 80% PB by own PM tertile; returns per-person table."""
    out = []
    for pid, g in ev.groupby("pid"):
        if pid not in persons:
            continue
        g = g.dropna(subset=["PM_sd", "pct"])
        if len(g) < 9:
            continue
        tert = pd.qcut(g.PM_sd.rank(method="first"), 3, labels=["low", "mid", "high"])
        cnt = tert.value_counts()
        if cnt.min() < 3:
            continue
        sh = (g.pct < 80).groupby(tert, observed=True).mean()
        out.append(dict(pid=pid, low=sh["low"], mid=sh["mid"], high=sh["high"], n=len(g)))
    return pd.DataFrame(out)


def l2_summary(t, rng):
    if len(t) == 0:
        return None
    idx = np.arange(len(t))
    bs = {k: [] for k in ["low", "mid", "high", "diff"]}
    for _ in range(NBOOT):
        s = t.iloc[rng.choice(idx, len(idx), replace=True)]
        for k in ["low", "mid", "high"]:
            bs[k].append(s[k].mean())
        bs["diff"].append((s.high - s.low).mean())
    r = dict(k=len(t), n=int(t.n.sum()))
    for k in ["low", "mid", "high"]:
        r[k] = float(t[k].mean() * 100); r[k + "_lo"] = float(np.percentile(bs[k], 2.5) * 100); r[k + "_hi"] = float(np.percentile(bs[k], 97.5) * 100)
    r["diff"] = float((t.high - t.low).mean() * 100)
    r["diff_lo"] = float(np.percentile(bs["diff"], 2.5) * 100); r["diff_hi"] = float(np.percentile(bs["diff"], 97.5) * 100)
    return r


def metareg(t, xcol):
    """WLS meta-regression of per-person slopes b0 on one covariate; weights 1/(se^2 + tau2_DL); HK-adjusted SE (floor 1)."""
    t = t.dropna(subset=["b0", "se0", xcol])
    k = len(t)
    if k < 4:
        return None
    tau2 = pool(t.b0, t.se0)["tau2"]
    w = 1 / (t.se0.to_numpy() ** 2 + tau2)
    X = np.column_stack([np.ones(k), t[xcol].to_numpy(float)])
    y = t.b0.to_numpy(float)
    XtW = X.T * w
    V = np.linalg.inv(XtW @ X)
    b = V @ XtW @ y
    r = y - X @ b
    q = float((w * r ** 2).sum() / (k - 2))
    se = math.sqrt(max(q, 1.0) * V[1, 1])
    tc = t_crit(k - 2)
    return dict(k=k, coef=float(b[1]), se=se, lo=float(b[1] - tc * se), hi=float(b[1] + tc * se), p=float(p_t(b[1] / se, k - 2)), tau2=float(tau2))


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pm-col", default="pm")
    ap.add_argument("--out", default="peakflow_paper/run_raw")
    a = ap.parse_args()
    out = os.path.join(ROOT, a.out); os.makedirs(out, exist_ok=True)
    rng = np.random.default_rng(SEED)
    d, cnt = load(a.pm_col)
    s, pb = select(d, "highest")
    R = Runner()
    desc = dict(counts=cnt, pm_col=a.pm_col, seed=SEED, nboot=NBOOT)

    ev = s[s.period == "evening"].copy()
    mo = s[s.period == "morning"].copy()
    desc["n_selected"] = dict(morning=int(len(mo)), evening=int(len(ev)), persons=int(s.pid.nunique()))
    multi = d.groupby(["pid", "workday", "period"]).size()
    desc["repeat_periods"] = dict(periods=int((multi > 1).sum()), dropped=int((multi - 1).sum()))
    s1pb = d.groupby("pid").personal_best.first()
    desc["pb_change_vs_s1_pct"] = {p: round(float((pb[p] / s1pb[p] - 1) * 100), 2) for p in pb.index}
    desc["F3_selected_rows"] = int(s.F3_poor_reproducibility.sum())
    desc["F4_selected_evening"] = int(ev.F4.sum())
    desc["evening_pct_below80"] = float((ev.pct < 80).mean() * 100)
    sdpct = ev.groupby("pid").pct.std()
    desc["within_sd_pct_ev"] = dict(median=float(sdpct.median()), q1=float(sdpct.quantile(.25)), q3=float(sdpct.quantile(.75)))
    sdpm = ev.dropna(subset=["PM_sd"]).groupby("pid").PM_sd.std()
    desc["within_sd_pm_ev_raw"] = dict(median=float(sdpm.median()), q1=float(sdpm.quantile(.25)), q3=float(sdpm.quantile(.75)))

    C = ["clock_u", "hours_on_shift_before_test"]
    # L1 family
    R.run("L1", ev, "pct", [("PM_sd", "z")], C, cities=True, label="L1 primary: %PB_ev ~ z(same-day mean PM) + clock + GPS hours on shift")
    R.run("L1_iqr", ev, "pct", [("PM_sd", "iqr")], C, label="L1 per within-rider IQR (rescale)")
    R.run("L1_raw10", ev, "pct", [("PM_sd", "raw10")], C, label="L1 per +10 raw units [CAL]")
    R.run("L1_cum", ev, "pct", [("PM_cum", "z")], C, label="L1-cum: z(same-day cumulative PM)")
    R.run("L1_T", ev, "pct", [("PM_sd", "z")], C + ["days_since_enrolment"], label="L1-T: + days since enrolment")
    # EQ in outcome-SD units
    t1 = R.persons[0]
    tsd = t1.assign(b0=t1.b0 / t1.sd_y, se0=t1.se0 / t1.sd_y)
    R.models["L1_sdunits"] = dict(label="L1 in outcome-SD units (per-person beta / SD of %PB)",
                                  **{"all:PM_sd:z": pool_rows(tsd), "slot20:PM_sd:z": pool_rows(tsd[tsd.slot20])})
    eq = {}
    for sset in ["all", "slot20"]:
        eq[sset] = dict(pp=tost(R.models["L1"][f"{sset}:PM_sd:z"], SESOI_PP),
                        sd=tost(R.models["L1_sdunits"][f"{sset}:PM_sd:z"], SESOI_SD))
    # L2
    l2 = {}
    for sset in ["all", "slot20"]:
        persons = set(ev.pid) if sset == "all" else set(ev[ev.slot20].pid)
        tt = boot_share_tertiles(ev, persons, rng)
        l2[sset] = l2_summary(tt, rng)
        if sset == "all":
            tt.to_csv(os.path.join(out, "l2_person_tertiles.csv"), index=False)
    # L3: within-day change
    m = mo[["pid", "workday", "pct", "PM_sd_any", "pm_sameday_nmin", "timestamp_local"]].rename(
        columns={"pct": "pct_mo", "PM_sd_any": "pm_mo", "pm_sameday_nmin": "nmin_mo", "timestamp_local": "t_mo"})
    pr = ev.merge(m, on=["pid", "workday"], how="inner")
    pr["delta"] = pr.pct - pr.pct_mo
    cumcol = f"{a.pm_col}_sameday_cum"
    if cumcol in d.columns:  # spec: (cum_ev - cum_mo) * 60 / (nmin_ev - nmin_mo); cum = sum of minute means / 60
        cmo = mo.set_index(["pid", "workday"])[cumcol]
        cm = pd.Series([cmo.get((p, w), np.nan) for p, w in zip(pr.pid, pr.workday)], index=pr.index).fillna(0) * 60
        ce = pr[cumcol] * 60
    else:  # calibrated columns have no cum: minute-weighted from the means
        cm = (pr.pm_mo.fillna(0) * pr.nmin_mo.fillna(0))
        ce = pr.PM_sd_any * pr.pm_sameday_nmin
    dn = pr.pm_sameday_nmin - pr.nmin_mo.fillna(0)
    pr["PM_between"] = ((ce - cm) / dn).where(dn >= 60)
    pr["hours_between"] = (pr.timestamp_local - pr.t_mo).dt.total_seconds() / 3600
    R.run("L3", pr, "delta", [("PM_between", "z")], ["hours_between"], label="L3: evening - morning %PB ~ z(PM between readings) + hours between")
    pr[["pid", "workday", "delta", "slot20", "city_c"]].to_csv(os.path.join(out, "l3_pairs.csv"), index=False)
    desc["l3_pairs"] = dict(n=int(len(pr)), with_pm_between=int(pr.PM_between.notna().sum()),
                            mean_delta=float(pr.delta.mean()), median_delta=float(pr.delta.median()))
    # L4 fatigue analogue
    R.run("L4a", ev, "pct", [("hours_on_shift_before_test", "none")], ["clock_u"], label="L4a: %PB_ev ~ GPS hours on shift + clock")
    R.run("L4b", ev, "pct", [("km10", "none")], ["clock_u"], label="L4b: %PB_ev ~ km ridden (per 10 km) + clock")
    R.run("L4c", ev.dropna(subset=["PM_sd"]), "pct", [("hours_on_shift_before_test", "none"), ("PM_sd", "z")], ["clock_u"], label="L4c: L4a + z(PM)")
    R.run("L4d", ev.dropna(subset=["PM_sd", "hours_on_shift_before_test"]), "pct", [("km10", "none"), ("PM_sd", "z")], ["clock_u"], label="L4d: L4b + z(PM)")
    R.run("L4a_samesample", ev.dropna(subset=["PM_sd"]), "pct", [("hours_on_shift_before_test", "none")], ["clock_u"], label="L4a on the L4c sample (for comparison)")
    R.run("L4b_samesample", ev.dropna(subset=["PM_sd", "hours_on_shift_before_test"]), "pct", [("km10", "none")], ["clock_u"], label="L4b on the L4d sample (for comparison)")
    # L5a last-entered value
    s_last, pb_last = select(d, "last")
    R.run("L5a", s_last[s_last.period == "evening"], "pct", [("PM_sd", "z")], C, label="L5a: last entered value per period (PB recomputed)")
    # L5b F4 indicator (only where it varies within the person)
    rows = []
    for pid, g in ev.groupby("pid"):
        ctr = C + (["F4"] if g.dropna(subset=["PM_sd"] + C).F4.nunique() > 1 else [])
        r = fit_person(g, "pct", [("PM_sd", "z")], ctr)
        if r:
            r.update(pid=pid, city=g.city_c.iloc[0], slot20=bool(g.slot20.iloc[0]), f4_term="F4" in ctr); rows.append(r)
    t5b = pd.DataFrame(rows)
    R.models["L5b"] = {"label": "L5b: L1 + F4 indicator (reading 0-2 min after a PVT; included where it varies)",
                       "all:PM_sd:z": pool_rows(t5b), "slot20:PM_sd:z": pool_rows(t5b[t5b.slot20]),
                       "persons_with_f4_term": int(t5b.f4_term.sum())}
    # L5d LOPO
    lo = []
    for p in t1.pid:
        r = pool_rows(t1[t1.pid != p]); lo.append(dict(drop=p, mu=r["mu"], p_hk=r["p_hk"], lo=r["lo_hk"], hi=r["hi_hk"]))
    lo = pd.DataFrame(lo); lo.to_csv(os.path.join(out, "l5d_lopo.csv"), index=False)
    lopo = dict(mu_min=float(lo.mu.min()), mu_max=float(lo.mu.max()), p_min=float(lo.p_hk.min()), p_max=float(lo.p_hk.max()),
                n_ci_excl0=int((lo.hi < 0).sum() + (lo.lo > 0).sum()), k=len(lo),
                drop_min_p=lo.loc[lo.p_hk.idxmin(), "drop"], drop_max_p=lo.loc[lo.p_hk.idxmax(), "drop"])
    # PB stability
    st = []
    for pid, g in s.groupby("pid"):
        g = g.sort_values("timestamp_local"); g = g[g.peak_flow >= 50]; h = len(g) // 2
        a1 = float(np.percentile(g.peak_flow.iloc[:h], 95)); a2 = float(np.percentile(g.peak_flow.iloc[h:], 95))
        st.append(dict(pid=pid, n=len(g), pb_first=a1, pb_second=a2, ratio=a2 / a1))
    st = pd.DataFrame(st); st.to_csv(os.path.join(out, "pb_stability.csv"), index=False)
    unstable = st[(st.ratio - 1).abs() > 0.10].pid.tolist()
    pbstab = dict(ratio_median=float(st.ratio.median()), ratio_min=float(st.ratio.min()), ratio_max=float(st.ratio.max()),
                  unstable=unstable, r12=float(st.set_index("pid").ratio.get("R12", np.nan)), r14=float(st.set_index("pid").ratio.get("R14", np.nan)))
    if unstable:
        R.run("L1_pbstable", ev[~ev.pid.isin(unstable)], "pct", [("PM_sd", "z")], C, label="L1 without persons whose PB halves differ > 10% (exploratory)")
    # L6 city difference
    L1m = R.models["L1"]
    bk, cmi = L1m["BKK:PM_sd:z"], L1m["CMI:PM_sd:z"]
    dd_ = cmi["mu"] - bk["mu"]; sed = math.sqrt(cmi["se_dl"] ** 2 + bk["se_dl"] ** 2); dfd = bk["k"] + cmi["k"] - 2
    l6 = dict(diff=dd_, se=sed, df=dfd, p=float(p_t(dd_ / sed, dfd)), lo=dd_ - t_crit(dfd) * sed, hi=dd_ + t_crit(dfd) * sed)
    for cc in ["BKK", "CMI"]:
        R.run(f"L1_{cc}_slot20", ev[(ev.city_c == cc) & ev.slot20], "pct", [("PM_sd", "z")], C, sets=("all",), label=f"L1 {cc}, 20-slot")
    # L7 placebo
    R.run("L7a", ev, "pct", [("PM_after", "z")], C, label="L7a: z(PM after the reading) in place of same-day PM")
    l7b_note = None
    if a.pm_col == "pm":
        e2 = pd.read_csv(os.path.join(ROOT, "calibration/tables/e2_person_days.csv"))
        e2["day_mean"] = (e2.sum_on_raw + e2.sum_off_raw) / (e2.n_on + e2.n_off)
        e2 = e2[(e2.n_on + e2.n_off) >= 60]
        e2["wd_lead"] = (pd.to_datetime(e2.workday) - pd.Timedelta(days=2)).dt.strftime("%Y-%m-%d")
        lead = e2[["person", "wd_lead", "day_mean"]].rename(columns={"person": "pid", "wd_lead": "workday", "day_mean": "PM_lead2"})
        evl = ev.merge(lead, on=["pid", "workday"], how="left")
        t7b, _ = R.run("L7b", evl, "pct", [("PM_lead2", "z")], C, label="L7b: z(PM two days later, S2 E2 person-days) in place of same-day PM")
        if len(t7b) < 5:
            l7b_note = f"infeasible: {len(t7b)} persons with >= {MIN_N} rows"
        desc["l7b_rows_with_lead2"] = int(evl.PM_lead2.notna().sum())
    else:
        l7b_note = "lead-2 PM exists only as raw in S2's tables; not run for this --pm-col"
    R.run("L7c", ev, "pct", [("PM_sd", "z"), ("PM_after", "z")], C, label="L7c: joint z(same-day) + z(after)")
    # S exposure sensitivities
    R.run("S_med", ev, "pct", [("PM_med", "z")], C, label="S-med: same-day median PM")
    R.run("S_clip", ev, "pct", [("PM_clip", "z")], C, label="S-clip: same-day clipped mean PM")
    R.run("S_onshift", ev, "pct", [("PM_on", "z")], C, label="S-onshift: on-shift mean PM (GPS shift start -> reading)")
    R.run("S_cov", ev, "pct", [("PM_sd_nocov", "z")], C, label="S-cov: no coverage rule")
    R.run("S_month", ev, "pct", [("PM_sd", "z")], C, dummies="month", label="S-month: + calendar-month dummies within person")
    # B2 / B3 Bangkok pilot
    demo = pd.read_csv(os.path.join(ROOT, "demographics/bkk_demographics.csv"))
    demo = demo[demo.pid.isin(pb.index)].copy(); demo["PB"] = demo.pid.map(pb)
    b2 = {}
    men, wom = demo[demo.sex == "M"], demo[demo.sex == "F"]
    ratio = men.PB.mean() / wom.PB.mean(); bs = []
    for _ in range(NBOOT):
        bs.append(men.PB.sample(len(men), replace=True, random_state=rng.integers(1 << 31)).mean() /
                  wom.PB.sample(len(wom), replace=True, random_state=rng.integers(1 << 31)).mean())
    b2["mf_ratio"] = dict(ratio=float(ratio), lo=float(np.percentile(bs, 2.5)), hi=float(np.percentile(bs, 97.5)), n_m=len(men), n_f=len(wom))
    age = demo.age_2026.to_numpy(float); pbv = demo.PB.to_numpy(float)
    def slope10(x, y):
        X = np.column_stack([np.ones(len(x)), x / 10]); return float(np.linalg.lstsq(X, y, rcond=None)[0][1])
    rho = float(pd.Series(age).rank().corr(pd.Series(pbv).rank()))
    bsl, brho = [], []
    for _ in range(NBOOT):
        i = rng.integers(0, len(age), len(age))
        if np.unique(age[i]).size < 2:
            continue
        bsl.append(slope10(age[i], pbv[i])); brho.append(pd.Series(age[i]).rank().corr(pd.Series(pbv[i]).rank()))
    b2["age"] = dict(slope_per10y=slope10(age, pbv), lo=float(np.percentile(bsl, 2.5)), hi=float(np.percentile(bsl, 97.5)),
                     spearman=rho, rho_lo=float(np.nanpercentile(brho, 2.5)), rho_hi=float(np.nanpercentile(brho, 97.5)), n=len(age))
    tb = t1[t1.city == "BKK"].merge(demo[["pid", "sex", "age_2026"]], on="pid")
    tb["age10"] = tb.age_2026 / 10; tb["male"] = (tb.sex == "M").astype(float)
    b3 = dict(age=metareg(tb, "age10"), sex_m_minus_f=metareg(tb, "male"), bkk_pooled=L1m["BKK:PM_sd:z"])
    fired = (b2["mf_ratio"]["ratio"] > 1) or (b2["age"]["slope_per10y"] < 0) or any(
        r and (r["lo"] > 0 or r["hi"] < 0) for r in [b3["age"], b3["sex_m_minus_f"]])
    b3["cmi_repeat_rule_fired"] = bool(fired)
    # write
    persons = pd.concat(R.persons, ignore_index=True)
    persons.to_csv(os.path.join(out, "per_person.csv"), index=False)
    allres = dict(models=R.models, eq=eq, l2=l2, lopo=lopo, pb_stability=pbstab, l6=l6, l7b_note=l7b_note,
                  b2=b2, b3=b3, desc=desc)
    with open(os.path.join(out, "pooled.json"), "w") as f:
        json.dump(allres, f, indent=1, default=lambda o: o if not isinstance(o, (np.floating, np.integer, np.bool_)) else o.item())
    print("written", out)


if __name__ == "__main__":
    main()
