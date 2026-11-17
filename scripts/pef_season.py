"""Paper 2, Addendum A (SPEC_ADDENDUM.md §11, approved 2026-11-12): season contrast between cities. Exploratory.

Run once:  python3 -I scripts/pef_season.py --pm-col pm
Within-rider change in mean %PB, haze (Mar 1 - Apr 30) - pre (Dec 12 - Feb 28); Chiang Mai - Bangkok.
Writes <out>/season.json, <out>/season_riders.csv, <out>/season_months.csv. numpy/pandas only.
"""
import argparse, itertools, json, math, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pef_stats import pool, p_t, t_crit  # noqa: E402
from pef_analysis import load, select, ROOT  # noqa: E402

SEED = 20261112
NPERM_SPEARMAN = 100000
PRE0, BOUND, HAZE1 = pd.Timestamp("2025-12-12"), pd.Timestamp("2026-03-01"), pd.Timestamp("2026-04-30")
UNSTABLE_PB = ["R05", "R13", "R15", "R20"]  # SPEC §4 PB stability result, listed in §11 A.2


def rider_deltas(x, minv, outcome="pct", shift_days=0, drop_first_days=0):
    x = x.copy()
    if drop_first_days:
        first = x.groupby("pid").wd.transform("min")
        x = x[x.wd >= first + pd.Timedelta(days=drop_first_days)]
    b = BOUND + pd.Timedelta(days=shift_days)
    x["win"] = np.where((x.wd >= PRE0) & (x.wd < b), "pre", np.where((x.wd >= b) & (x.wd <= HAZE1), "haze", ""))
    x = x[x.win != ""]
    if outcome == "below80":
        x["y"] = (x.pct < 80).astype(float) * 100
    else:
        x["y"] = x.pct
    rows = []
    for pid, g in x.groupby("pid"):
        a, h = g[g.win == "pre"].y, g[g.win == "haze"].y
        if len(a) < minv or len(h) < minv:
            continue
        se = math.sqrt(a.var(ddof=1) / len(a) + h.var(ddof=1) / len(h))
        pm_a, pm_h = g[g.win == "pre"].PM_sd.mean(), g[g.win == "haze"].PM_sd.mean()
        rows.append(dict(pid=pid, city=g.city_c.iloc[0], slot20=bool(g.slot20.iloc[0]), n_pre=len(a), n_haze=len(h),
                         mean_pre=a.mean(), mean_haze=h.mean(), delta=h.mean() - a.mean(), se=se,
                         pm_ratio=pm_h / pm_a if pm_a > 0 else np.nan))
    return pd.DataFrame(rows)


def contrast(t):
    """CMI - BKK: exact permutation p and HKSJ-style CI."""
    c, b = t[t.city == "CMI"], t[t.city == "BKK"]
    obs = c.delta.mean() - b.delta.mean()
    v = t.delta.to_numpy(); n = len(v); kc = len(c); tot = v.sum()
    cnt = 0; m = 0
    for idx in itertools.combinations(range(n), kc):
        sc = v[list(idx)].sum()
        st = sc / kc - (tot - sc) / (n - kc)
        cnt += abs(st) >= abs(obs) - 1e-12; m += 1
    base = dict(k_cmi=kc, k_bkk=len(b), mean_delta_cmi=float(c.delta.mean()), mean_delta_bkk=float(b.delta.mean()),
                diff_unweighted=float(obs), p_perm=cnt / m, n_perm=m)
    if (t.se < 1e-3).any():  # a rider with no within-window variance (e.g. 0% below 80 in both windows): SE 0, DL weights undefined
        base["ci_note"] = f"HKSJ-style CI not estimable: {int((t.se < 1e-3).sum())} riders have zero within-window variance"
        return base
    pc, pb = pool(c.delta, c.se), pool(b.delta, b.se)
    d = pc["mu"] - pb["mu"]; se = math.sqrt(pc["se_hk"] ** 2 + pb["se_hk"] ** 2); df = n - 2
    tc = t_crit(df)
    return dict(**base,
                cmi_dl=pc["mu"], cmi_lo=pc["lo_hk"], cmi_hi=pc["hi_hk"], bkk_dl=pb["mu"], bkk_lo=pb["lo_hk"], bkk_hi=pb["hi_hk"],
                diff_dl=d, diff_lo=d - tc * se, diff_hi=d + tc * se, p_t=float(p_t(d / se, df)), df=df)


def spearman(t, rng):
    t = t.dropna(subset=["pm_ratio"])
    r1, r2 = t.delta.rank(), t.pm_ratio.rank()
    rho = float(r1.corr(r2))
    a = r1.to_numpy(); b = r2.to_numpy(); cnt = 0
    for _ in range(NPERM_SPEARMAN):
        cnt += abs(np.corrcoef(rng.permutation(a), b)[0, 1]) >= abs(rho) - 1e-12
    return dict(k=len(t), rho=rho, p_perm=(cnt + 1) / (NPERM_SPEARMAN + 1))


def holm(ps):
    order = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0.0
    for i, j in enumerate(order):
        run = max(run, min(1.0, (m - i) * ps[j])); adj[j] = run
    return adj.tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pm-col", default="pm")
    ap.add_argument("--out", default="peakflow_paper/run_season")
    a = ap.parse_args()
    out = os.path.join(ROOT, a.out); os.makedirs(out, exist_ok=True)
    rng = np.random.default_rng(SEED)
    d, _ = load(a.pm_col)
    s, _ = select(d, "highest")
    s["wd"] = pd.to_datetime(s.workday)
    ev, mo = s[s.period == "evening"], s[s.period == "morning"]
    res = {}
    t_ev = rider_deltas(ev, 5); t_mo = rider_deltas(mo, 5)
    t_ev.to_csv(os.path.join(out, "season_riders.csv"), index=False)
    res["T1_evening"] = contrast(t_ev)
    res["T2_morning"] = contrast(t_mo)
    res["T3_spearman"] = spearman(t_ev, rng)
    ps = [res["T1_evening"]["p_perm"], res["T2_morning"]["p_perm"], res["T3_spearman"]["p_perm"]]
    res["holm"] = dict(zip(["T1", "T2", "T3"], holm(ps)))
    r1 = res["T1_evening"]
    res["wording"] = ("consistent with a seasonal association (exploratory)"
                      if res["holm"]["T1"] < 0.05 and r1["diff_dl"] < 0 and r1["diff_unweighted"] < 0
                      else "not consistent with a seasonal association at this sample size")
    sens = {}
    sens["S1_drop_first_2wk"] = contrast(rider_deltas(ev, 5, drop_first_days=14))
    sens["S2_no_unstable_pb"] = contrast(t_ev[~t_ev.pid.isin(UNSTABLE_PB)])
    sens["S3a_shift_minus_2wk"] = contrast(rider_deltas(ev, 5, shift_days=-14))
    sens["S3b_shift_plus_2wk"] = contrast(rider_deltas(ev, 5, shift_days=14))
    sens["S4_share_below80"] = contrast(rider_deltas(ev, 5, outcome="below80"))
    sens["S5_min8"] = contrast(rider_deltas(ev, 8))
    res["sensitivity"] = sens
    res["slot20_equals_primary"] = bool(t_ev.slot20.all())
    res["riders"] = dict(BKK=sorted(t_ev[t_ev.city == "BKK"].pid), CMI=sorted(t_ev[t_ev.city == "CMI"].pid))
    # monthly means for the figure (riders in the primary set)
    m = ev[ev.pid.isin(t_ev.pid)].copy(); m["month"] = m.wd.dt.strftime("%Y-%m")
    m.groupby(["pid", "city_c", "month"]).pct.agg(["mean", "size"]).reset_index().to_csv(os.path.join(out, "season_months.csv"), index=False)
    with open(os.path.join(out, "season.json"), "w") as f:
        json.dump(res, f, indent=1, default=float)
    print("written", out)


if __name__ == "__main__":
    main()
