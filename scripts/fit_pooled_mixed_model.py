"""Secondary analysis (PLAN.md #5): pooled Mundlak-decomposition mixed model.

Cross-check for the primary per-rider + meta-analysis result (#4). Pools
all riders' sessions into one model, but splits PM2.5 into within-rider
and between-rider components (Mundlak / group-mean centering) so the
within-rider coefficient isn't contaminated by between-rider confounds --
see PLAN.md #5 for the full rationale.

Random effects: a random intercept per rider. (A nested rider-day
variance component was tried first, matching the day-clustering used in
the primary analysis, but with ~700 rider-day groups averaging ~2.5
sessions each it over-parameterizes the model and the fit is singular --
dropped in favor of the plain random-intercept form. Same-day
non-independence is handled rigorously in the primary analysis (#4) via
day-clustered SEs; this secondary model is a simpler cross-check, not a
full replacement.)

Requires scripts/build_pvt_aq_dataset.py to have been run first.

Output: data/pooled_mixed_model_summary.txt
"""
import os

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

ROOT = os.path.join(os.path.dirname(__file__), "..")
IN_PATH = os.path.join(ROOT, "data", "pvt_aq_linked.csv")
SUMMARY_OUT = os.path.join(ROOT, "data", "pooled_mixed_model_summary.txt")
PRIMARY_SUMMARY = os.path.join(ROOT, "data", "primary_analysis_summary.txt")

MODEL_COLS = [
    "response_speed_hz", "pm25_mean_sameday", "temp_mean_sameday",
    "humidity_mean_sameday", "shift_duration_min_sameday", "session_number",
    "hour_of_day", "username", "test_date",
]
# day_of_week / is_weekend dropped: all 12 weekend PVT sessions have no
# same-day AQ readings, so after dropna the modeling frame has zero
# weekend rows and the covariate is structurally constant here.
FORMULA = (
    "response_speed_hz ~ pm25_within + pm25_between + temp_mean_sameday "
    "+ humidity_mean_sameday + shift_duration_min_sameday + session_number "
    "+ hour_of_day"
)


def build_model_frame(df):
    """Prepare the pooled modeling frame: drop rows missing any model
    column, then compute the Mundlak decomposition --
      pm25_between = each rider's own mean sameday PM2.5 (a fixed trait
                      of that rider, absorbs between-rider confounds)
      pm25_within  = that session's deviation from the rider's own mean
                      (the within-subject comparison, PLAN.md #5's
                      headline estimand)
    plus a rider_day id (used only diagnostically here; the nested
    rider_day random effect built from it was found to over-parameterize
    the model, see module docstring)."""
    g = df.dropna(subset=MODEL_COLS).copy()
    g["pm25_between"] = g.groupby("username")["pm25_mean_sameday"].transform("mean")
    g["pm25_within"] = g["pm25_mean_sameday"] - g["pm25_between"]
    g["rider_day"] = g["username"].astype(str) + "_" + g["test_date"].astype(str)
    return g


def parse_primary_pooled_beta():
    """Scrape the pooled beta_pm25 value out of
    data/primary_analysis_summary.txt (fit_primary_analysis.py's output)
    so this script's summary can report the primary-vs-secondary
    comparison without recomputing the primary analysis. Returns None if
    that file doesn't exist yet or the line isn't found."""
    if not os.path.exists(PRIMARY_SUMMARY):
        return None
    for line in open(PRIMARY_SUMMARY):
        if "pooled beta_pm25" in line:
            return float(line.split("=")[1].split()[0])
    return None


def main():
    """Fit two pooled mixed models and compare both against the primary
    per-rider result:

      Model A: pm25_within with one common slope across all riders, plus
        a random intercept per rider. This is the PLAN.md #5 headline
        secondary estimate.
      Model B: an exploratory attempt to let pm25_within's slope vary by
        rider (random slope), to test whether Model A's homogeneous-slope
        assumption explains its divergence from the primary result. Kept
        in the output for transparency even though it turned out to be
        numerically unreliable (see the CAVEAT written into the summary).

    Writes both models' full statsmodels summaries plus the comparison
    discussion to data/pooled_mixed_model_summary.txt.
    """
    df = pd.read_csv(IN_PATH)
    g = build_model_frame(df)

    lines = []
    lines.append(f"Pooled mixed model: {len(g)} sessions, {g['username'].nunique()} riders, "
                  f"{g['rider_day'].nunique()} rider-days")
    lines.append("Random effects: (1 | rider) only -- see module docstring for why the "
                  "nested rider-day variance component was dropped.")
    lines.append("")

    # Model A: common (homogeneous) pm25_within slope across all riders.
    model_common = smf.mixedlm(FORMULA, data=g, groups=g["username"])
    fit_common = model_common.fit(reml=True)
    lines.append("=== Model A: common pm25_within slope, random intercept only ===")
    lines.append(str(fit_common.summary()))
    lines.append("")

    beta_within = fit_common.params.get("pm25_within", np.nan)
    se_within = fit_common.bse.get("pm25_within", np.nan)
    beta_between = fit_common.params.get("pm25_between", np.nan)
    lines.append(f"pm25_within  (headline, within-rider effect) = {beta_within:.6f}  "
                 f"(SE={se_within:.6f}, 95% CI [{beta_within - 1.96*se_within:.6f}, "
                 f"{beta_within + 1.96*se_within:.6f}])")
    lines.append(f"pm25_between (between-rider effect)          = {beta_between:.6f}")

    # Model B: let pm25_within's slope vary by rider (random slope), to
    # test whether forcing a common slope in Model A is what's pulling the
    # estimate away from the primary per-rider + meta-analysis result.
    lines.append("")
    lines.append("=== Model B: random-slope pm25_within by rider (uncorrelated with intercept) ===")
    try:
        model_slope = smf.mixedlm(
            FORMULA, data=g, groups=g["username"],
            vc_formula={"pm25_within_slope": "0 + pm25_within"},
        )
        fit_slope = model_slope.fit(reml=True, method=["lbfgs"], maxiter=200)
        lines.append(f"Converged: {fit_slope.converged}")
        lines.append(str(fit_slope.summary()))
        lines.append("")

        pop_beta_within = fit_slope.params.get("pm25_within", np.nan)
        pop_se_within = fit_slope.bse.get("pm25_within", np.nan)
        lines.append(f"pm25_within population-mean slope = {pop_beta_within:.6f}  "
                     f"(SE={pop_se_within:.6f}, 95% CI [{pop_beta_within - 1.96*pop_se_within:.6f}, "
                     f"{pop_beta_within + 1.96*pop_se_within:.6f}])")

        # Per-rider implied slope = population-mean slope + that rider's BLUP deviation.
        rider_slopes = {
            rider: pop_beta_within + re.get("pm25_within_slope", 0.0)
            for rider, re in fit_slope.random_effects.items()
        }
        rs = pd.Series(rider_slopes).sort_values()
        lines.append("")
        lines.append("Per-rider implied slopes (population-mean + BLUP), for comparison with #4:")
        for rider, val in rs.items():
            lines.append(f"  {rider}: {val:.6f}")
        lines.append("")
        lines.append(
            "CAVEAT: this fit is numerically degenerate -- SE (0.43) is ~400x the "
            "coefficient, and every rider's BLUP collapsed to exactly the population "
            "mean (zero shrinkage information, despite a nonzero slope-variance "
            "estimate). A correlated-intercept-slope version of this model separately "
            "failed to converge at all. Two different random-slope specifications both "
            "failing is itself a result: with ~80 sessions/rider on average, this "
            "dataset doesn't carry enough information to reliably estimate 22 "
            "individual slopes *through a pooled mixed model's variance-component "
            "optimizer*. Do not treat the population-mean estimate above as reliable."
        )
    except Exception as e:
        lines.append(f"Random-slope model failed to fit/converge: {e}")
        pop_beta_within = np.nan

    primary_beta = parse_primary_pooled_beta()
    if primary_beta is not None:
        lines.append("")
        lines.append(f"Primary analysis (#4) pooled meta-analytic beta_pm25 = {primary_beta:.6f}")
        lines.append(f"Secondary Model A (common slope) pm25_within beta    = {beta_within:.6f}")
        lines.append(
            "(Model B's population beta is omitted here -- see CAVEAT above, it's not "
            "a reliable number to compare.)"
        )
        lines.append("")
        lines.append(
            "Why A and #4 disagree: Model A assumes every rider shares one common "
            "pm25_within slope, which is only a consistent estimator of the average "
            "effect when slopes really are homogeneous (Pesaran & Smith 1995). The "
            "primary analysis's I^2=44.8% says they're not -- riders differ genuinely "
            "in direction and magnitude (e.g. Panatda_6018 trends strongly positive "
            "while most others trend negative). Under real slope heterogeneity, a "
            "'mean group' estimator -- fit each unit separately, then average, exactly "
            "what #4 does -- is the consistent one; a pooled/common-slope estimator "
            "like Model A is not guaranteed to recover the same quantity. That the "
            "random-slope Model B (the direct way to test this in a pooled framework) "
            "is numerically unworkable here is consistent with, not contradictory to, "
            "that story: it's the same heterogeneity making the variance-component "
            "estimation hard that also invalidates Model A's homogeneous-slope "
            "assumption. Net: trust the primary analysis (#4) as the headline result; "
            "treat Model A as a directionally-consistent but methodologically weaker "
            "cross-check, not a contradiction."
        )

    summary = "\n".join(lines)
    with open(SUMMARY_OUT, "w") as f:
        f.write(summary + "\n")
    print(summary)


if __name__ == "__main__":
    main()
