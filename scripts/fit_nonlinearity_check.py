"""Robustness check (PLAN.md #6): is the PM2.5-performance relationship
actually linear, or is a threshold/saturation effect being missed?

Consistent with the primary analysis (§4) and unlike an earlier version of
this script, this fits a quadratic term *per rider* and meta-analyzes it
-- not a pooled/common-effect model. A per-rider quadratic term needs
only one extra parameter (vs. e.g. 3 for quartile dummies), which is
feasible even at ~90 sessions/rider; quartile dummies were not, which is
why an earlier version of this check fell back to pooling across riders
-- the same pooling that diluted §5's Model A relative to the primary
per-rider result. This version avoids that trade-off entirely.

The quadratic term uses pm25_within (rider-mean-centered exposure)
rather than raw pm25_mean_sameday, since centering around zero is what
keeps the linear and quadratic terms from being highly collinear.

Requires scripts/build_pvt_aq_dataset.py to have been run first.

Output: data/nonlinearity_check_summary.txt, data/nonlinearity_check_forest.png
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = ["DejaVu Sans", "Noto Sans Thai"]
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from fit_primary_analysis import (
    random_effects_meta, make_forest_plot,
    IN_PATH, MIN_SESSIONS_PER_RIDER, MIN_DAYS_FOR_RELIABLE_CLUSTERING,
)

ROOT = os.path.join(os.path.dirname(__file__), "..")
SUMMARY_OUT = os.path.join(ROOT, "data", "nonlinearity_check_summary.txt")
PER_RIDER_OUT = os.path.join(ROOT, "data", "nonlinearity_check_per_rider.csv")
FOREST_OUT = os.path.join(ROOT, "data", "nonlinearity_check_forest.png")

MODEL_COLS = [
    "response_speed_hz", "pm25_mean_sameday", "temp_mean_sameday",
    "humidity_mean_sameday", "shift_duration_min_sameday", "session_number",
    "hour_of_day", "username",
]
FORMULA = (
    "response_speed_hz ~ pm25_within + I(pm25_within ** 2) + temp_mean_sameday "
    "+ humidity_mean_sameday + shift_duration_min_sameday + session_number "
    "+ hour_of_day"
)
QUAD_TERM = "I(pm25_within ** 2)"


def build_frame(df):
    g = df.dropna(subset=MODEL_COLS).copy()
    g["pm25_between"] = g.groupby("username")["pm25_mean_sameday"].transform("mean")
    g["pm25_within"] = g["pm25_mean_sameday"] - g["pm25_between"]
    return g


def fit_per_rider_quadratic(df):
    """Same per-rider + day-clustered-SE approach as fit_primary_analysis,
    but extracting the quadratic term's coefficient instead of the linear
    PM2.5 term."""
    rows = []
    for username, g in df.groupby("username"):
        n_days = g["test_date"].nunique()
        if len(g) < MIN_SESSIONS_PER_RIDER:
            rows.append({"username": username, "n": len(g), "n_days": n_days, "included": False,
                         "reason": "too few complete sessions"})
            continue
        try:
            fit = smf.ols(FORMULA, data=g).fit(
                cov_type="cluster", cov_kwds={"groups": g["test_date"]}
            )
        except Exception as e:
            rows.append({"username": username, "n": len(g), "n_days": n_days, "included": False,
                         "reason": f"fit failed: {e}"})
            continue
        beta = fit.params.get(QUAD_TERM, np.nan)
        se = fit.bse.get(QUAD_TERM, np.nan)
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


def main():
    df = pd.read_csv(IN_PATH)
    g = build_frame(df)

    per_rider = fit_per_rider_quadratic(g)
    per_rider.to_csv(PER_RIDER_OUT, index=False)

    included = per_rider[per_rider["included"]]
    excluded = per_rider[~per_rider["included"]]

    lines = []
    lines.append("Nonlinearity check: per-rider quadratic term, meta-analyzed (consistent")
    lines.append("with the primary §4 per-rider methodology, not pooled).")
    lines.append(f"Formula per rider: {FORMULA}")
    lines.append(f"Quadratic term uses pm25_within (rider-mean-centered) to avoid")
    lines.append(f"linear/quadratic collinearity.")
    lines.append("")
    lines.append(f"Per-rider models fit: {len(included)}/{len(per_rider)} riders included")
    if len(excluded):
        lines.append("Excluded:")
        for _, r in excluded.iterrows():
            lines.append(f"  {r['username']}: n={r['n']}, reason: {r['reason']}")

    thin_clusters = included[included["n_days"] < MIN_DAYS_FOR_RELIABLE_CLUSTERING]
    if len(thin_clusters):
        lines.append(
            f"Caveat: {len(thin_clusters)} rider(s) have <{MIN_DAYS_FOR_RELIABLE_CLUSTERING} distinct "
            "days, cluster-robust SEs less reliable (flagged, not excluded):"
        )
        for _, r in thin_clusters.iterrows():
            lines.append(f"  {r['username']}: n_days={r['n_days']}")
    lines.append("")

    if len(included) < 2:
        lines.append("Fewer than 2 riders with valid fits -- cannot run meta-analysis.")
    else:
        meta = random_effects_meta(included["beta_pm25"].to_numpy(), included["se_pm25"].to_numpy())
        lines.append("Random-effects meta-analysis of the quadratic term (DerSimonian-Laird):")
        lines.append(f"  k (riders)          = {meta['k']}")
        lines.append(f"  pooled quad. beta    = {meta['mu_re']:.8f}")
        lines.append(f"  SE                   = {meta['se_re']:.8f}")
        lines.append(f"  95% CI               = [{meta['ci_low']:.8f}, {meta['ci_high']:.8f}]")
        lines.append(f"  z                    = {meta['z']:.3f}")
        lines.append(f"  I^2 (% het.)         = {meta['i2']:.1f}%")
        lines.append("")
        sig = meta["ci_low"] > 0 or meta["ci_high"] < 0
        lines.append(
            f"Interpretation: {'quadratic term is significant -- evidence of curvature/nonlinearity' if sig else 'CI includes zero -- no evidence of curvature; linear model remains a reasonable fit'}."
        )
        make_forest_plot(
            per_rider, meta, out_path=FOREST_OUT,
            xlabel="Quadratic PM2.5 term coefficient (I(pm25_within**2)) -- nonzero implies curvature",
            title="Per-rider PM2.5 quadratic term\nwith random-effects pooled estimate",
        )
        lines.append(f"\nForest plot written to {FOREST_OUT}")

    summary = "\n".join(lines)
    with open(SUMMARY_OUT, "w") as f:
        f.write(summary + "\n")
    print(summary)


if __name__ == "__main__":
    main()
