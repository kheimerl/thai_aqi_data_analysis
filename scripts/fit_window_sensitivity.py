"""Robustness check (PLAN.md #6): exposure-window sensitivity.

Re-runs the exact same per-rider + random-effects meta-analysis pipeline
as the primary analysis (fit_primary_analysis.py), but swaps the primary
same-day cumulative PM2.5 exposure variable for the 1h and 4h trailing
windows. If PVT performance tracks cumulative exposure rather than just
recent air (the hypothesis behind making same-day primary in PLAN.md #3),
the same-day estimate should be at least as stable/significant as the
short trailing windows, not an artifact of one arbitrary window choice.

Requires scripts/build_pvt_aq_dataset.py and fit_primary_analysis.py to
have been run first (this script re-reads pvt_aq_linked.csv directly and
also re-parses the sameday result already written by
primary_analysis_summary.txt for the comparison table).

Outputs:
  data/window_sensitivity_summary.txt
  data/window_sensitivity_forest_1h.png
  data/window_sensitivity_forest_4h.png
"""
import os
import re

import pandas as pd

from fit_primary_analysis import (
    fit_per_rider, random_effects_meta, make_forest_plot,
    SUMMARY_OUT as PRIMARY_SUMMARY, IN_PATH,
)

ROOT = os.path.join(os.path.dirname(__file__), "..")
OUT_SUMMARY = os.path.join(ROOT, "data", "window_sensitivity_summary.txt")

WINDOWS = {
    "sameday": dict(exposure_col="pm25_mean_sameday", label="same-day cumulative (primary)"),
    "1h": dict(exposure_col="pm25_mean_1h", label="1h trailing window"),
    "4h": dict(exposure_col="pm25_mean_4h", label="4h trailing window"),
}


def parse_primary_summary():
    """Pull the already-computed sameday pooled result out of
    primary_analysis_summary.txt instead of re-running it, so this script
    only has to do the new 1h/4h work."""
    if not os.path.exists(PRIMARY_SUMMARY):
        return None
    text = open(PRIMARY_SUMMARY).read()
    fields = {}
    for key, pattern in [
        ("k", r"k \(riders\)\s*=\s*(\d+)"),
        ("mu_re", r"pooled beta_pm25\s*=\s*([-\d.eE]+)"),
        ("se_re", r"SE\s*=\s*([-\d.eE]+)"),
        ("ci_low", r"95% CI\s*=\s*\[([-\d.eE]+),"),
        ("ci_high", r"95% CI\s*=\s*\[[-\d.eE]+,\s*([-\d.eE]+)\]"),
        ("z", r"z\s*=\s*([-\d.eE]+)"),
        ("i2", r"I\^2 \(% het\.\)\s*=\s*([-\d.eE]+)"),
    ]:
        m = re.search(pattern, text)
        fields[key] = float(m.group(1)) if m else None
    return fields


def main():
    df = pd.read_csv(IN_PATH)

    lines = []
    lines.append("Exposure-window sensitivity (PLAN.md #6)")
    lines.append("Same per-rider + random-effects meta-analysis pipeline as the primary")
    lines.append("analysis, run against each candidate exposure window.")
    lines.append("")

    results = {}
    sameday = parse_primary_summary()
    if sameday is not None:
        results["sameday"] = sameday
        lines.append(f"sameday (primary, already computed): beta={sameday['mu_re']:.6f}, "
                     f"95% CI=[{sameday['ci_low']:.6f}, {sameday['ci_high']:.6f}], "
                     f"z={sameday['z']:.3f}, I2={sameday['i2']:.1f}%")
    else:
        lines.append("WARNING: could not parse primary_analysis_summary.txt for the sameday result.")

    for wname in ["1h", "4h"]:
        spec = WINDOWS[wname]
        per_rider = fit_per_rider(df, exposure_col=spec["exposure_col"])
        included = per_rider[per_rider["included"]]
        lines.append("")
        lines.append(f"=== {wname} ({spec['label']}) ===")
        lines.append(f"Per-rider models fit: {len(included)}/{len(per_rider)} riders included")
        if len(included) < 2:
            lines.append("Fewer than 2 riders with valid fits -- cannot run meta-analysis.")
            continue
        meta = random_effects_meta(included["beta_pm25"].to_numpy(), included["se_pm25"].to_numpy())
        results[wname] = meta
        lines.append(f"  pooled beta = {meta['mu_re']:.6f}  SE={meta['se_re']:.6f}  "
                     f"95% CI=[{meta['ci_low']:.6f}, {meta['ci_high']:.6f}]  "
                     f"z={meta['z']:.3f}  I2={meta['i2']:.1f}%")
        out_path = os.path.join(ROOT, "data", f"window_sensitivity_forest_{wname}.png")
        make_forest_plot(
            per_rider, meta, out_path=out_path,
            xlabel=f"PM2.5 coefficient on response speed (Hz per µg/m³), {spec['label']}",
        )
        lines.append(f"  Forest plot: {out_path}")

    lines.append("")
    lines.append("Comparison:")
    lines.append(f"{'window':<10}{'beta':>12}{'CI_low':>12}{'CI_high':>12}{'z':>8}{'I2%':>8}")
    for wname in ["sameday", "1h", "4h"]:
        r = results.get(wname)
        if r is None:
            lines.append(f"{wname:<10}  (not available)")
            continue
        lines.append(f"{wname:<10}{r['mu_re']:>12.6f}{r['ci_low']:>12.6f}{r['ci_high']:>12.6f}"
                     f"{r['z']:>8.3f}{r['i2']:>8.1f}")

    summary = "\n".join(lines)
    with open(OUT_SUMMARY, "w") as f:
        f.write(summary + "\n")
    print(summary)


if __name__ == "__main__":
    main()
