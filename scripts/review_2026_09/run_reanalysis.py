"""Run the analyses declared in REANALYSIS_SPEC_2026-09-28.md on linked_corrected.csv."""
import _paths  # side effect: chdir into data/review_2026_09/
import sys
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

from _paths import ROOT as REPO
import fit_primary_analysis as P  # noqa: E402
from fit_tau2_estimator_check import hksj_adjust, reml_tau2  # noqa: E402

warnings.filterwarnings("ignore")

IN = sys.argv[1] if len(sys.argv) > 1 else "linked_corrected.csv"


def cr2_se(fit, groups):
    """Bell-McCaffrey CR2 cluster-robust SEs for an OLS fit."""
    X = fit.model.exog
    e = fit.resid.to_numpy()
    bread = np.linalg.pinv(X.T @ X)
    meat = np.zeros_like(bread)
    g = np.asarray(groups)
    for c in np.unique(g):
        idx = np.where(g == c)[0]
        Xc, ec = X[idx], e[idx]
        M = np.eye(len(idx)) - Xc @ bread @ Xc.T
        vals, vecs = np.linalg.eigh((M + M.T) / 2)
        inv_sqrt = np.where(vals > 1e-10, 1 / np.sqrt(np.clip(vals, 1e-10, None)), 0.0)
        A = (vecs * inv_sqrt) @ vecs.T
        u = Xc.T @ (A @ ec)
        meat += np.outer(u, u)
    V = bread @ meat @ bread
    return pd.Series(np.sqrt(np.clip(np.diag(V), 0, None)), index=fit.params.index)


def per_rider(df, outcome, exposures, covars, se_type="CR2"):
    """Per-rider OLS; returns rows with beta/se for every exposure term."""
    rhs = " + ".join(exposures + covars)
    formula = f"{outcome} ~ {rhs}"
    cols = [outcome] + exposures + covars
    rows = []
    for u, g in df.groupby("username"):
        g = g.dropna(subset=cols)
        if len(g) < P.MIN_SESSIONS_PER_RIDER:
            continue
        fit = smf.ols(formula, data=g).fit()
        if se_type == "CR1":
            se = smf.ols(formula, data=g).fit(cov_type="cluster", cov_kwds={"groups": g["study_day"]}).bse
        else:
            se = cr2_se(fit, g["study_day"])
        r = {"username": u, "n": len(g), "n_days": g.study_day.nunique()}
        for x in exposures:
            r[f"b_{x}"], r[f"se_{x}"] = fit.params[x], se[x]
        rows.append(r)
    return pd.DataFrame(rows)


def pool(beta, se, method="REML+HKSJ"):
    beta, se = np.asarray(beta), np.asarray(se)
    ok = np.isfinite(beta) & np.isfinite(se) & (se > 0)
    beta, se = beta[ok], se[ok]
    if method == "DL+Wald":
        m = P.random_effects_meta(beta, se)
        p = 2 * stats.norm.sf(abs(m["z"]))
        return dict(k=m["k"], mu=m["mu_re"], lo=m["ci_low"], hi=m["ci_high"], p=p, i2=m["i2"])
    tau2 = reml_tau2(beta, se)[0]
    h = hksj_adjust(beta, se, tau2)
    p = 2 * stats.t.sf(abs(h["t"]), h["df"])
    tau2_dl, q, qdf = P.dl_tau2(beta, se)
    i2 = max(0.0, (q - qdf) / q) * 100 if q > 0 else 0.0
    return dict(k=len(beta), mu=h["mu"], lo=h["ci_low"], hi=h["ci_high"], p=p, i2=i2)


def fmt(label, r, scale=1e4):
    return (f"{label:<58s} k={r['k']:2d}  beta={r['mu']*scale:+7.2f}  95%CI=[{r['lo']*scale:+7.2f}, {r['hi']*scale:+7.2f}]"
            f"  p={r['p']:.3f}  I2={r['i2']:4.1f}%")


def main():
    df = pd.read_csv(IN)
    df["test_date"] = df["study_day"]  # clustering unit for the repo's fit_per_rider
    base_cov = ["temp_mean_sameday", "humidity_mean_sameday", "shift_duration_min_sameday",
                "session_number", "hour_of_day"]
    Y, Yc = "response_speed_hz", "response_speed_clean_hz"
    print(f"input={IN}  (beta and CI shown x1e4, i.e. Hz per 10,000 ug/m3; original headline = -6.41)\n")

    # ---- R0: plan of record, code of record, mechanical fixes only
    pr = P.fit_per_rider(df)
    inc = pr[pr.included]
    r0 = pool(inc.beta_pm25, inc.se_pm25, "DL+Wald")
    print(fmt("R0  plan-of-record (CR1, DL, Wald) [repo code]", r0))

    # ---- R1: inference-corrected
    t = per_rider(df, Y, ["pm25_mean_sameday"], base_cov[:2] + base_cov[2:])
    r1 = pool(t.b_pm25_mean_sameday, t.se_pm25_mean_sameday)
    print(fmt("R1  CR2 + REML + HKSJ   <-- headline", r1))
    t_cr1 = per_rider(df, Y, ["pm25_mean_sameday"], base_cov, se_type="CR1")
    print(fmt("    (R1 per-rider fits, CR1 SEs, REML+HKSJ)", pool(t_cr1.b_pm25_mean_sameday, t_cr1.se_pm25_mean_sameday)))
    print(fmt("    (R1 per-rider fits, CR2 SEs, DL+Wald)", pool(t.b_pm25_mean_sameday, t.se_pm25_mean_sameday, "DL+Wald")))
    mg = t.b_pm25_mean_sameday
    tt = stats.ttest_1samp(mg, 0)
    print(f"    unweighted mean-group: beta={mg.mean()*1e4:+.2f}  t={tt.statistic:.2f}  p={tt.pvalue:.3f}  "
          f"riders negative: {(mg<0).sum()}/{len(mg)}")

    # ---- R2: false starts removed
    t2 = per_rider(df, Yc, ["pm25_mean_sameday"], base_cov)
    print(fmt("R2  R1 + RT<100ms false starts removed", pool(t2.b_pm25_mean_sameday, t2.se_pm25_mean_sameday)))

    # ---- S1: correctly aligned trailing windows
    for w in ("1h", "4h"):
        cov = [f"temp_mean_{w}", f"humidity_mean_{w}", "shift_duration_min_sameday", "session_number", "hour_of_day"]
        tw = per_rider(df, Y, [f"pm25_mean_{w}"], cov)
        print(fmt(f"S1  trailing {w} (window-matched temp/RH)", pool(tw[f"b_pm25_mean_{w}"], tw[f"se_pm25_mean_{w}"])))

    # ---- S2: movement-onset -> test
    cov = ["temp_mean_onset", "humidity_mean_onset", "onset_to_test_min", "session_number", "hour_of_day"]
    ts2 = per_rider(df, Y, ["pm25_mean_onset"], cov)
    print(fmt("S2  movement-onset -> test", pool(ts2.b_pm25_mean_onset, ts2.se_pm25_mean_onset)))

    # ---- N1: negative control, post-test air
    cov = ["temp_mean_post4h", "humidity_mean_post4h", "shift_duration_min_sameday", "session_number", "hour_of_day"]
    tn = per_rider(df, Y, ["pm25_mean_post4h"], cov)
    print(fmt("N1  NEGATIVE CONTROL: post-test (test, test+4h]", pool(tn.b_pm25_mean_post4h, tn.se_pm25_mean_post4h)))
    cov = ["temp_mean_4h", "humidity_mean_4h", "shift_duration_min_sameday", "session_number", "hour_of_day"]
    tj = per_rider(df, Y, ["pm25_mean_4h", "pm25_mean_post4h"], cov)
    print(fmt("N1j joint model: pre-test 4h term", pool(tj.b_pm25_mean_4h, tj.se_pm25_mean_4h)))
    print(fmt("N1j joint model: post-test 4h term", pool(tj.b_pm25_mean_post4h, tj.se_pm25_mean_post4h)))

    # ---- N2: pooled, rider FE + region x study-day FE
    d = df.dropna(subset=[Y, "pm25_mean_sameday", "temp_mean_sameday", "humidity_mean_sameday",
                          "session_number", "hour_of_day"]).copy()
    d["region_day"] = d.region.astype(str) + "_" + d.study_day.astype(str)
    d["day_code"] = pd.factorize(d.study_day)[0]
    d["rider_code"] = pd.factorize(d.username)[0]
    rhs = "pm25_mean_sameday + temp_mean_sameday + humidity_mean_sameday + session_number + hour_of_day + C(username)"
    for label, f in (("N2a pooled, rider FE only", rhs), ("N2b pooled, rider FE + region x day FE", rhs + " + C(region_day)")):
        m = smf.ols(f"{Y} ~ {f}", data=d)
        for ct, kw in (("day-clustered", {"groups": d.day_code}),
                       ("2-way rider x day", {"groups": np.column_stack([d.rider_code, d.day_code])})):
            fit = m.fit(cov_type="cluster", cov_kwds=kw)
            b, lo, hi = fit.params["pm25_mean_sameday"], *fit.conf_int().loc["pm25_mean_sameday"]
            print(f"{label + ' [' + ct + ']':<58s} n={len(d)}  beta={b*1e4:+7.2f}  95%CI=[{lo*1e4:+7.2f}, {hi*1e4:+7.2f}]"
                  f"  p={fit.pvalues['pm25_mean_sameday']:.3f}")
    # How much PM2.5 variance survives the FE?
    fe = smf.ols("pm25_mean_sameday ~ C(username) + C(region_day)", data=d).fit()
    rid = smf.ols("pm25_mean_sameday ~ C(username)", data=d).fit()
    print(f"    within-rider PM2.5 SD={rid.resid.std():.1f} ug/m3; after region x day FE SD={fe.resid.std():.1f} ug/m3 "
          f"(share of within-rider variance left: {fe.resid.var()/rid.resid.var():.2f})")


if __name__ == "__main__":
    main()
