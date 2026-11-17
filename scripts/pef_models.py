"""Paper 2, Addendum D (SPEC_ADDENDUM.md §14, 2026-11-17): model-based versions of L1. Exploratory.

Run once:  python3 -I scripts/pef_models.py --pm-col pm
Pooled OLS, rider fixed effects, Mundlak within/between, FE + time, placebo; cluster-robust SE by rider (CR1, t G-1 df).
numpy/pandas only. Writes peakflow_paper/run_models/models.json.
"""
import argparse, json, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pef_stats import p_t, t_crit  # noqa: E402
from pef_analysis import load, select, ROOT  # noqa: E402


def crfit(y, X, g, names, absorbed=0):
    """OLS with CR1 cluster-robust SE by g; t with G-1 df. absorbed = number of absorbed FE (dof)."""
    n, p = X.shape
    XtXi = np.linalg.pinv(X.T @ X); b = XtXi @ X.T @ y; u = y - X @ b
    G = len(np.unique(g)); meat = np.zeros((p, p))
    for k in np.unique(g):
        m = g == k; s = X[m].T @ u[m]; meat += np.outer(s, s)
    c = G / (G - 1) * (n - 1) / max(n - p - absorbed, 1)
    se = np.sqrt(np.diag(c * XtXi @ meat @ XtXi)); tc = t_crit(G - 1)
    return {nm: dict(b=float(b[i]), se=float(se[i]), lo=float(b[i] - tc * se[i]), hi=float(b[i] + tc * se[i]),
                     p=float(p_t(b[i] / se[i], G - 1))) for i, nm in enumerate(names) if nm}, dict(n=int(n), G=int(G))


def demean(df, cols, by="pid"):
    return df[cols] - df.groupby(by)[cols].transform("mean")


def prep(df, y, x, ctrls, min_rows=2):
    df = df.dropna(subset=[y, x] + ctrls).copy()
    df = df[df.groupby("pid")[y].transform("size") >= min_rows]
    sdw = float(df.groupby("pid")[x].std().median())
    df["xsd"] = df[x] / sdw
    df["xz"] = df.groupby("pid")[x].transform(lambda v: (v - v.mean()) / v.std(ddof=1))
    return df, sdw


def fe(df, y, xcol, ctrls, extra_dummies=None):
    cols = [y, xcol] + ctrls
    D = demean(df, cols)
    X = [D[xcol].to_numpy(float)] + [D[c].to_numpy(float) for c in ctrls]; names = ["x"] + [None] * len(ctrls)
    if extra_dummies is not None:
        dm = pd.get_dummies(df[extra_dummies], drop_first=True, dtype=float)
        dm = dm - dm.groupby(df.pid).transform("mean")
        X += [dm[c].to_numpy() for c in dm.columns]; names += [None] * dm.shape[1]
    X = np.column_stack(X)
    keep = np.linalg.matrix_rank(X) == X.shape[1]
    r, info = crfit(D[y].to_numpy(float), X, df.pid.to_numpy(), names, absorbed=df.pid.nunique())
    info["full_rank"] = bool(keep)
    return dict(**r["x"], **info)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--pm-col", default="pm")
    ap.add_argument("--out", default="peakflow_paper/run_models"); a = ap.parse_args()
    out = os.path.join(ROOT, a.out); os.makedirs(out, exist_ok=True)
    d, _ = load(a.pm_col); s, _ = select(d, "highest")
    ev = s[s.period == "evening"].copy(); mo = s[s.period == "morning"].copy()
    C = ["clock_u", "hours_on_shift_before_test"]
    R = {}
    df, sdw = prep(ev, "pct", "PM_sd", C); R["sdw_raw"] = sdw
    g = df.pid.to_numpy()
    # (a) pooled OLS
    X = np.column_stack([np.ones(len(df)), df["xsd"]] + [df[c] for c in C])
    r, info = crfit(df.pct.to_numpy(float), X, g, [None, "x", None, None]); R["a_pooled"] = dict(**r["x"], **info)
    # (b) FE, typical-SD scale and per-rider z scale
    R["b_fe"] = fe(df, "pct", "xsd", C)
    R["b_fe_z"] = fe(df, "pct", "xz", C)
    # (c) Mundlak
    df["x_mean"] = df.groupby("pid")["xsd"].transform("mean"); df["x_within"] = df["xsd"] - df.x_mean
    X = np.column_stack([np.ones(len(df)), df.x_within, df.x_mean] + [df[c] for c in C])
    r, info = crfit(df.pct.to_numpy(float), X, g, [None, "within", "between", None, None])
    R["c_mundlak"] = dict(within=r["within"], between=r["between"], **info)
    # (d) FE + days since enrolment + month
    R["d_fe_time"] = fe(df, "pct", "xsd", C + ["days_since_enrolment"], extra_dummies="month")
    # (f) placebo PM after, in (b)
    dfa, sda = prep(ev, "pct", "PM_after", C); R["f_placebo_fe"] = fe(dfa, "pct", "xsd", C); R["sdw_after_raw"] = sda
    # (b) without riders who repeat values
    R["b_fe_no_R09_R14_R20"] = fe(df[~df.pid.isin(["R09", "R14", "R20"])], "pct", "xsd", C)
    # (b) for the evening - morning drop (L3 sample)
    m = mo[["pid", "workday", "pct", "pm_sameday", "pm_sameday_nmin", "pm_sameday_cum", "timestamp_local"]].rename(
        columns=lambda c: c if c in ("pid", "workday") else c + "_mo")
    pr = ev.merge(m, on=["pid", "workday"])
    pr["delta"] = pr.pct - pr.pct_mo
    dn = pr.pm_sameday_nmin - pr.pm_sameday_nmin_mo.fillna(0)
    pr["PM_between"] = ((pr.pm_sameday_cum * 60 - pr.pm_sameday_cum_mo.fillna(0) * 60) / dn).where(dn >= 60) if a.pm_col == "pm" else np.nan
    pr["hours_between"] = (pr.timestamp_local - pr.timestamp_local_mo).dt.total_seconds() / 3600
    dfd, _ = prep(pr, "delta", "PM_between", ["hours_between"]); R["b_fe_drop"] = fe(dfd, "delta", "xsd", ["hours_between"])
    # (b) for night air -> morning (spec §12 sample)
    mo["PM_night"] = mo[f"{a.pm_col}_night00"].where(mo.pm_night00_nmin >= 180)
    mo = mo[mo.pm_after24_nmin >= 180]
    dfn, sdn = prep(mo, "pct", "PM_night", ["days_since_enrolment", "clock_u"])
    R["b_fe_night"] = fe(dfn, "pct", "xsd", ["days_since_enrolment", "clock_u"]); R["sdw_night_raw"] = sdn
    # descriptive: why pooled can differ — between-rider correlation of mean PM and mean %PB
    rm = df.groupby("pid")[["PM_sd", "pct"]].mean()
    R["between_corr_meanPM_meanPctPB"] = float(rm.PM_sd.corr(rm.pct)); R["between_k"] = int(len(rm))
    with open(os.path.join(out, "models.json"), "w") as f:
        json.dump(R, f, indent=1)
    print("written", out)


if __name__ == "__main__":
    main()
