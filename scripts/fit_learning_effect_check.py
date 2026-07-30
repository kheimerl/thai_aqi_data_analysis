"""Robustness check (PLAN.md #6): does dropping each rider's first week
of PVT sessions change the primary result?

The primary model already controls for practice/learning effects via a
linear session_number term (PLAN.md #4/#5). But PVT learning effects are
typically front-loaded -- most of the improvement happens in the first
few sessions, then flattens out -- so a linear term may not fully
capture the shape of it, and the earliest, noisiest, most-affected
sessions could be distorting each rider's regression more than a linear
control removes.

This re-runs the exact same per-rider + random-effects meta-analysis
pipeline as the primary analysis (fit_primary_analysis.py), but on a
copy of the data with each rider's first calendar week (from that
rider's own first test date) excluded. If the pooled estimate is stable
after dropping these sessions, that's evidence the linear session_number
control was already doing its job; a meaningfully different estimate
would suggest early-session noise/learning was contaminating the primary
result.

Requires scripts/build_pvt_aq_dataset.py and fit_primary_analysis.py to
have been run first (re-reads pvt_aq_linked.csv directly and re-parses
the already-computed full-sample result from primary_analysis_summary.txt
for the comparison table).

Outputs:
  data/learning_effect_check_summary.txt
  data/learning_effect_check_forest.png
  data/learning_effect_check_per_rider.csv
"""
import os

import pandas as pd

from fit_primary_analysis import (
    fit_per_rider, random_effects_meta, make_forest_plot, IN_PATH,
)
from fit_window_sensitivity import parse_primary_summary

ROOT = os.path.join(os.path.dirname(__file__), "..")
OUT_SUMMARY = os.path.join(ROOT, "data", "learning_effect_check_summary.txt")
OUT_PER_RIDER = os.path.join(ROOT, "data", "learning_effect_check_per_rider.csv")
FOREST_OUT = os.path.join(ROOT, "data", "learning_effect_check_forest.png")

FIRST_WEEK_DAYS = 7


def drop_first_week(df):
    """Return a copy of df with each rider's sessions from their own first
    FIRST_WEEK_DAYS calendar days (relative to that rider's own first test
    date, not a shared study-wide date) removed. Uses test_date (already a
    calendar date, not a timestamp) so this is a day-granularity cutoff."""
    df = df.copy()
    df["test_date"] = pd.to_datetime(df["test_date"])
    starts = df.groupby("username")["test_date"].transform("min")
    keep = df["test_date"] >= starts + pd.Timedelta(days=FIRST_WEEK_DAYS)
    return df[keep].copy()


def main():
    df = pd.read_csv(IN_PATH)
    df["test_date"] = pd.to_datetime(df["test_date"])

    filtered = drop_first_week(df)

    lines = []
    lines.append(f"Learning-effect robustness check: drop each rider's first {FIRST_WEEK_DAYS} "
                 "calendar days of PVT sessions (relative to their own start date), then re-run "
                 "the primary per-rider + meta-analysis pipeline.")
    lines.append("")

    dropped_by_rider = (
        df.groupby("username").size() - filtered.groupby("username").size()
    ).fillna(df.groupby("username").size()).astype(int)
    total_before, total_after = len(df), len(filtered)
    lines.append(f"Sessions before: {total_before}  after: {total_after}  "
                 f"dropped: {total_before - total_after} "
                 f"({100 * (total_before - total_after) / total_before:.1f}%)")
    lines.append("Dropped sessions per rider:")
    for u, n in dropped_by_rider.sort_values(ascending=False).items():
        pct = 100 * n / df[df["username"] == u].shape[0]
        lines.append(f"  {u}: {n} ({pct:.1f}%)")
    lines.append("")

    per_rider = fit_per_rider(filtered)
    per_rider.to_csv(OUT_PER_RIDER, index=False)
    included = per_rider[per_rider["included"]]
    excluded = per_rider[~per_rider["included"]]

    lines.append(f"Per-rider models fit (first week dropped): {len(included)}/{len(per_rider)} "
                 "riders included")
    if len(excluded):
        lines.append("Excluded:")
        for _, r in excluded.iterrows():
            lines.append(f"  {r['username']}: n={r['n']}, reason: {r['reason']}")
    lines.append("")

    primary = parse_primary_summary()

    if len(included) < 2:
        lines.append("Fewer than 2 riders with valid fits -- cannot run meta-analysis.")
    else:
        meta = random_effects_meta(included["beta_pm25"].to_numpy(), included["se_pm25"].to_numpy())
        lines.append(f"Random-effects meta-analysis (first week dropped, DerSimonian-Laird):")
        lines.append(f"  k (riders)        = {meta['k']}")
        lines.append(f"  pooled beta_pm25  = {meta['mu_re']:.6f}")
        lines.append(f"  SE                = {meta['se_re']:.6f}")
        lines.append(f"  95% CI            = [{meta['ci_low']:.6f}, {meta['ci_high']:.6f}]")
        lines.append(f"  z                 = {meta['z']:.3f}")
        lines.append(f"  I^2 (% het.)      = {meta['i2']:.1f}%")
        lines.append("")

        if primary is not None:
            lines.append("Comparison with primary analysis (full sample, first week included):")
            lines.append(f"  full sample:        beta={primary['mu_re']:.6f}  "
                         f"95% CI=[{primary['ci_low']:.6f}, {primary['ci_high']:.6f}]  "
                         f"z={primary['z']:.3f}  I2={primary['i2']:.1f}%")
            lines.append(f"  first week dropped: beta={meta['mu_re']:.6f}  "
                         f"95% CI=[{meta['ci_low']:.6f}, {meta['ci_high']:.6f}]  "
                         f"z={meta['z']:.3f}  I2={meta['i2']:.1f}%")
            diff = meta["mu_re"] - primary["mu_re"]
            lines.append(f"  difference (dropped - full): {diff:.6f}")
            lines.append("")

            magnitude_stable = abs(diff) < 0.5 * abs(primary["mu_re"])
            primary_sig = primary["ci_low"] > 0 or primary["ci_high"] < 0
            dropped_sig = meta["ci_low"] > 0 or meta["ci_high"] < 0
            sig_flipped = primary_sig and not dropped_sig

            lines.append(
                "Interpretation: " + (
                    "point estimate magnitude is similar (change is less than half the original "
                    "effect size), consistent with the linear session_number term already "
                    "adequately controlling for practice/learning effects."
                    if magnitude_stable else
                    "point estimate shifts substantially in magnitude after dropping the first "
                    "week -- the linear session_number control may not be fully capturing "
                    "front-loaded learning effects."
                )
            )
            if sig_flipped:
                lines.append(
                    "CAVEAT: despite the similar point estimate, the result crosses the "
                    "significance boundary -- the full-sample estimate was significant "
                    f"(95% CI excludes 0: [{primary['ci_low']:.6f}, {primary['ci_high']:.6f}]) "
                    "but the first-week-dropped estimate is not "
                    f"(95% CI includes 0: [{meta['ci_low']:.6f}, {meta['ci_high']:.6f}]). "
                    "This is expected with ~7.5% less data and a wider CI even under a truly "
                    "stable effect, but it means this check does not independently confirm "
                    "significance -- treat the primary analysis's significance as somewhat "
                    "less robust than it would look in isolation."
                )
        else:
            lines.append("WARNING: could not parse primary_analysis_summary.txt for comparison.")

        make_forest_plot(
            per_rider, meta, out_path=FOREST_OUT,
            xlabel=f"PM2.5 coefficient on response speed (Hz per µg/m³), first {FIRST_WEEK_DAYS} days dropped",
            title="Per-rider PM2.5 effect, first week excluded\nwith random-effects pooled estimate",
        )
        lines.append(f"\nForest plot written to {FOREST_OUT}")

    summary = "\n".join(lines)
    with open(OUT_SUMMARY, "w") as f:
        f.write(summary + "\n")
    print(summary)


if __name__ == "__main__":
    main()
