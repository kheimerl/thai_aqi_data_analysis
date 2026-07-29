"""Primary analysis (PLAN.md #4): per-rider PM2.5 regression, combined via
random-effects meta-analysis.

Step 1: for each rider, fit an independent OLS regression of PVT response
speed on same-day cumulative PM2.5 exposure plus covariates, using only
that rider's own sessions.

Step 2: combine the per-rider PM2.5 coefficients via a DerSimonian-Laird
random-effects meta-analysis, weighting each rider's estimate by its
precision rather than averaging them naively.

The core per-rider + meta-analysis logic (fit_per_rider, random_effects_meta)
is reused by fit_window_sensitivity.py (PLAN.md #6) to re-run this same
analysis against the 1h/4h trailing-window exposure variables instead of
the primary same-day one.

Requires scripts/build_pvt_aq_dataset.py to have been run first.

Outputs:
  data/primary_analysis_per_rider.csv   -- one row per rider's fit
  data/primary_analysis_summary.txt     -- pooled estimate + heterogeneity
  data/primary_analysis_forest.png      -- forest plot
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# A rider's PVT username is in Thai script; the default font lacks Thai
# glyphs, so fall back to Noto Sans Thai (installed on this system) for it.
plt.rcParams["font.family"] = ["DejaVu Sans", "Noto Sans Thai"]
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

ROOT = os.path.join(os.path.dirname(__file__), "..")
IN_PATH = os.path.join(ROOT, "data", "pvt_aq_linked.csv")
PER_RIDER_OUT = os.path.join(ROOT, "data", "primary_analysis_per_rider.csv")
SUMMARY_OUT = os.path.join(ROOT, "data", "primary_analysis_summary.txt")
FOREST_OUT = os.path.join(ROOT, "data", "primary_analysis_forest.png")

EXPOSURE_COL = "pm25_mean_sameday"
TEMP_COL = "temp_mean_sameday"
HUMIDITY_COL = "humidity_mean_sameday"
# shift_duration_min only exists for the sameday window (it's meaningless
# for a fixed-length trailing window) -- reused as-is for the window
# sensitivity checks, since "time on shift so far" is the same confound to
# control for regardless of which exposure window is being tested.
SHIFT_DURATION_COL = "shift_duration_min_sameday"

MIN_SESSIONS_PER_RIDER = 15
# Below this many distinct days, cluster-robust SEs are unreliable
# (too few clusters for the asymptotics) -- flag, don't exclude.
MIN_DAYS_FOR_RELIABLE_CLUSTERING = 20


def formula_for(exposure_col, temp_col, humidity_col):
    # day_of_week / is_weekend dropped: all 12 weekend PVT sessions in the
    # raw data have no same-day AQ readings (sensors appear to be off on
    # weekends), so after dropna the modeling frame has zero weekend rows --
    # the covariate is structurally constant here and was causing a
    # singular design matrix downstream (see fit_pooled_mixed_model.py).
    return (
        f"response_speed_hz ~ {exposure_col} + {temp_col} + {humidity_col} "
        f"+ {SHIFT_DURATION_COL} + session_number + hour_of_day"
    )


def fit_per_rider(df, exposure_col=EXPOSURE_COL, temp_col=TEMP_COL, humidity_col=HUMIDITY_COL):
    """Fit each rider's own OLS with standard errors clustered by test_date.

    Riders often take 2-3 PVT sessions per shift; same-day sessions share
    that day's exposure trajectory and likely correlated unobserved
    factors (fatigue, mood), so treating them as independent observations
    understates the SEs. Clustering by day corrects for that -- it leaves
    beta unchanged but widens SE for riders whose sessions cluster heavily
    within days, which the meta-analysis step will down-weight accordingly.
    """
    formula = formula_for(exposure_col, temp_col, humidity_col)
    model_cols = [
        "response_speed_hz", exposure_col, temp_col, humidity_col,
        SHIFT_DURATION_COL, "session_number", "hour_of_day",
    ]
    rows = []
    for username, g in df.groupby("username"):
        g = g.dropna(subset=model_cols)
        n_days = g["test_date"].nunique()
        if len(g) < MIN_SESSIONS_PER_RIDER:
            rows.append({"username": username, "n": len(g), "n_days": n_days, "included": False,
                         "reason": "too few complete sessions"})
            continue
        try:
            fit = smf.ols(formula, data=g).fit(
                cov_type="cluster", cov_kwds={"groups": g["test_date"]}
            )
        except Exception as e:  # rank-deficient / singular design
            rows.append({"username": username, "n": len(g), "n_days": n_days, "included": False,
                         "reason": f"fit failed: {e}"})
            continue
        beta = fit.params.get(exposure_col, np.nan)
        se = fit.bse.get(exposure_col, np.nan)
        if not np.isfinite(beta) or not np.isfinite(se) or se <= 0:
            rows.append({"username": username, "n": len(g), "n_days": n_days, "included": False,
                         "reason": "non-finite coefficient/SE (likely collinear)"})
            continue
        rows.append({
            "username": username, "n": len(g), "n_days": n_days, "included": True, "reason": "",
            "beta_pm25": beta, "se_pm25": se,
            "ci_low": beta - 1.96 * se, "ci_high": beta + 1.96 * se,
            "r_squared": fit.rsquared,
        })
    return pd.DataFrame(rows)


def random_effects_meta(beta, se):
    """DerSimonian-Laird random-effects meta-analysis."""
    v = se ** 2
    w_fe = 1.0 / v
    mu_fe = np.sum(w_fe * beta) / np.sum(w_fe)
    q = np.sum(w_fe * (beta - mu_fe) ** 2)
    k = len(beta)
    df = k - 1
    c = np.sum(w_fe) - np.sum(w_fe ** 2) / np.sum(w_fe)
    tau2 = max(0.0, (q - df) / c) if c > 0 else 0.0
    i2 = max(0.0, (q - df) / q) * 100 if q > 0 else 0.0

    w_re = 1.0 / (v + tau2)
    mu_re = np.sum(w_re * beta) / np.sum(w_re)
    se_re = np.sqrt(1.0 / np.sum(w_re))
    return {
        "k": k, "mu_re": mu_re, "se_re": se_re,
        "ci_low": mu_re - 1.96 * se_re, "ci_high": mu_re + 1.96 * se_re,
        "z": mu_re / se_re, "tau2": tau2, "i2": i2, "q": q, "q_df": df,
    }


def make_forest_plot(per_rider, meta, out_path=FOREST_OUT,
                      xlabel="PM2.5 coefficient on response speed (Hz per µg/m³), same-day cumulative exposure",
                      title="Per-rider PM2.5 effect on PVT response speed\nwith random-effects pooled estimate"):
    included = per_rider[per_rider["included"]].sort_values("beta_pm25")
    fig_height = 0.35 * len(included) + 2
    fig, ax = plt.subplots(figsize=(8, fig_height))

    y = np.arange(len(included))
    ax.errorbar(
        included["beta_pm25"], y,
        xerr=[included["beta_pm25"] - included["ci_low"], included["ci_high"] - included["beta_pm25"]],
        fmt="o", color="#3b6fa0", ecolor="#8fa8bf", elinewidth=1.5, capsize=3, markersize=5,
    )
    ax.axvline(0, color="#999999", linewidth=1, linestyle="--")

    pooled_y = -1.5
    ax.errorbar(
        [meta["mu_re"]], [pooled_y],
        xerr=[[meta["mu_re"] - meta["ci_low"]], [meta["ci_high"] - meta["mu_re"]]],
        fmt="D", color="#c0392b", ecolor="#c0392b", elinewidth=2, capsize=4, markersize=8,
    )

    yticks = list(y) + [pooled_y]
    yticklabels = [
        f"{u}  (n={n}, days={d})"
        for u, n, d in zip(included["username"], included["n"], included["n_days"])
    ]
    yticklabels.append(f"Pooled (random-effects, k={meta['k']})")
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels, fontsize=8)
    ax.set_ylim(pooled_y - 1, len(included))

    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_title(title, fontsize=12)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    df = pd.read_csv(IN_PATH)

    per_rider = fit_per_rider(df)
    per_rider.to_csv(PER_RIDER_OUT, index=False)

    included = per_rider[per_rider["included"]]
    excluded = per_rider[~per_rider["included"]]

    lines = []
    lines.append(f"Per-rider models fit: {len(included)}/{len(per_rider)} riders included")
    lines.append("SEs are clustered by test_date (riders take 2-3 PVT sessions/shift).")
    if len(excluded):
        lines.append("Excluded:")
        for _, r in excluded.iterrows():
            lines.append(f"  {r['username']}: n={r['n']}, reason: {r['reason']}")

    thin_clusters = included[included["n_days"] < MIN_DAYS_FOR_RELIABLE_CLUSTERING]
    if len(thin_clusters):
        lines.append(
            f"Caveat: {len(thin_clusters)} rider(s) have <{MIN_DAYS_FOR_RELIABLE_CLUSTERING} distinct "
            "days, where cluster-robust SEs are less reliable (not excluded, just flagged):"
        )
        for _, r in thin_clusters.iterrows():
            lines.append(f"  {r['username']}: n_days={r['n_days']}")
    lines.append("")

    if len(included) < 2:
        lines.append("Fewer than 2 riders with valid fits -- cannot run meta-analysis.")
    else:
        meta = random_effects_meta(included["beta_pm25"].to_numpy(), included["se_pm25"].to_numpy())
        lines.append("Random-effects meta-analysis (DerSimonian-Laird):")
        lines.append(f"  k (riders)        = {meta['k']}")
        lines.append(f"  pooled beta_pm25  = {meta['mu_re']:.6f}  (Hz per ug/m3, same-day mean PM2.5)")
        lines.append(f"  SE                = {meta['se_re']:.6f}")
        lines.append(f"  95% CI            = [{meta['ci_low']:.6f}, {meta['ci_high']:.6f}]")
        lines.append(f"  z                 = {meta['z']:.3f}")
        lines.append(f"  tau^2 (het. var.) = {meta['tau2']:.8f}")
        lines.append(f"  I^2 (% het.)      = {meta['i2']:.1f}%")
        lines.append(f"  Q ({meta['q_df']} df)       = {meta['q']:.3f}")
        make_forest_plot(per_rider, meta)
        lines.append(f"\nForest plot written to {FOREST_OUT}")

    summary = "\n".join(lines)
    with open(SUMMARY_OUT, "w") as f:
        f.write(summary + "\n")
    print(summary)


if __name__ == "__main__":
    main()
