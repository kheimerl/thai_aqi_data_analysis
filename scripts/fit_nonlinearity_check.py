"""Robustness check (PLAN.md #6): is the PM2.5-performance relationship
actually linear, or is the linear primary/secondary models' assumption
hiding a threshold/saturation effect?

Per-rider quartile binning (each rider's own sessions split into Q1-Q4 by
their own PM2.5_within distribution -- preserves the within-subjects
framing, same logic as the primary analysis) would leave only ~20-25
sessions per rider per bin, too little to fit reliably per rider. So
this check is done at the pooled level (more data to work with), same
structure as the PLAN.md #5 secondary model (Mundlak decomposition,
random intercept per rider) but with PM2.5 entered as quartile dummies
instead of a continuous linear term, plus an AIC comparison against the
continuous version.

Requires scripts/build_pvt_aq_dataset.py to have been run first.

Output: data/nonlinearity_check_summary.txt, data/nonlinearity_check.png
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = ["DejaVu Sans", "Noto Sans Thai"]
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

ROOT = os.path.join(os.path.dirname(__file__), "..")
IN_PATH = os.path.join(ROOT, "data", "pvt_aq_linked.csv")
SUMMARY_OUT = os.path.join(ROOT, "data", "nonlinearity_check_summary.txt")
PLOT_OUT = os.path.join(ROOT, "data", "nonlinearity_check.png")

MODEL_COLS = [
    "response_speed_hz", "pm25_mean_sameday", "temp_mean_sameday",
    "humidity_mean_sameday", "shift_duration_min_sameday", "session_number",
    "hour_of_day", "username",
]
LINEAR_FORMULA = (
    "response_speed_hz ~ pm25_within + pm25_between + temp_mean_sameday "
    "+ humidity_mean_sameday + shift_duration_min_sameday + session_number "
    "+ hour_of_day"
)
QUARTILE_FORMULA = (
    "response_speed_hz ~ C(pm25_quartile) + pm25_between + temp_mean_sameday "
    "+ humidity_mean_sameday + shift_duration_min_sameday + session_number "
    "+ hour_of_day"
)


def build_model_frame(df):
    g = df.dropna(subset=MODEL_COLS).copy()
    g["pm25_between"] = g.groupby("username")["pm25_mean_sameday"].transform("mean")
    g["pm25_within"] = g["pm25_mean_sameday"] - g["pm25_between"]
    # Per-rider quartiles of that rider's own exposure -- a within-subjects
    # comparison, same logic as the linear pm25_within term, just binned.
    g["pm25_quartile"] = g.groupby("username")["pm25_within"].transform(
        lambda s: pd.qcut(s, 4, labels=["Q1", "Q2", "Q3", "Q4"], duplicates="drop")
    )
    return g


def main():
    df = pd.read_csv(IN_PATH)
    g = build_model_frame(df)
    g = g.dropna(subset=["pm25_quartile"])

    lines = []
    lines.append(f"Nonlinearity check: {len(g)} sessions, {g['username'].nunique()} riders")
    lines.append("Per-rider quartiles of within-rider PM2.5 exposure (Q1=cleanest, Q4=dirtiest,")
    lines.append("relative to that rider's own distribution), pooled model with (1 | rider).")
    lines.append("")

    fit_linear = smf.mixedlm(LINEAR_FORMULA, data=g, groups=g["username"]).fit(reml=False)
    fit_quart = smf.mixedlm(QUARTILE_FORMULA, data=g, groups=g["username"]).fit(reml=False)

    lines.append(f"Linear model AIC    = {fit_linear.aic:.2f}")
    lines.append(f"Quartile model AIC  = {fit_quart.aic:.2f}")
    lines.append(f"(Lower AIC is a better fit; a much lower quartile-model AIC would suggest")
    lines.append(f" real nonlinearity the linear term misses. Compared with REML=False/ML")
    lines.append(f" fitting since AIC comparisons across differently-parameterized fixed")
    lines.append(f" effects require ML, not REML.)")
    lines.append("")

    lines.append("Quartile coefficients (relative to Q1, the cleanest-air quartile):")
    quartile_effects = {"Q1": 0.0}
    quartile_ci = {"Q1": (0.0, 0.0)}
    for q in ["Q2", "Q3", "Q4"]:
        key = f"C(pm25_quartile)[T.{q}]"
        beta = fit_quart.params.get(key, np.nan)
        se = fit_quart.bse.get(key, np.nan)
        quartile_effects[q] = beta
        quartile_ci[q] = (beta - 1.96 * se, beta + 1.96 * se)
        lines.append(f"  {q}: beta={beta:.6f}  SE={se:.6f}  95% CI=[{beta - 1.96*se:.6f}, {beta + 1.96*se:.6f}]")

    vals = [quartile_effects[q] for q in ["Q1", "Q2", "Q3", "Q4"]]
    is_monotonic = all(vals[i] >= vals[i + 1] for i in range(len(vals) - 1)) or \
                   all(vals[i] <= vals[i + 1] for i in range(len(vals) - 1))
    lines.append("")
    lines.append(f"Monotonic Q1->Q4 trend: {is_monotonic} "
                 f"({'consistent with a smooth dose-response' if is_monotonic else 'NOT monotonic -- possible threshold/non-monotonic effect'})")
    lines.append(
        "CAVEAT: like PLAN.md #5's Model A, this check pools all riders under one "
        "common effect (here, one common quartile pattern) rather than letting each "
        "rider have their own shape -- the same pooling that diluted #5's estimate "
        "relative to the primary per-rider analysis. A null/non-monotonic result here "
        "means 'no evidence of nonlinearity in the pooled/average sense', not a strong "
        "claim that no individual rider has a nonlinear dose-response -- testing that "
        "properly would need per-rider quartile fits, which aren't reliable at "
        "~20-25 sessions per rider per quartile bin."
    )

    fig, ax = plt.subplots(figsize=(6, 4))
    qs = ["Q1", "Q2", "Q3", "Q4"]
    y = [quartile_effects[q] for q in qs]
    yerr_lo = [quartile_effects[q] - quartile_ci[q][0] for q in qs]
    yerr_hi = [quartile_ci[q][1] - quartile_effects[q] for q in qs]
    ax.errorbar(qs, y, yerr=[yerr_lo, yerr_hi], fmt="o-", color="#3b6fa0",
                ecolor="#8fa8bf", elinewidth=1.5, capsize=4, markersize=6)
    ax.axhline(0, color="#999999", linewidth=1, linestyle="--")
    ax.set_xlabel("Within-rider PM2.5 quartile (Q1=cleanest, Q4=dirtiest)")
    ax.set_ylabel("Response speed effect vs. Q1 (Hz)")
    ax.set_title("PM2.5 dose-response by quartile\n(pooled model, relative to each rider's own Q1)")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(PLOT_OUT, dpi=150)
    plt.close(fig)
    lines.append(f"\nPlot written to {PLOT_OUT}")

    summary = "\n".join(lines)
    with open(SUMMARY_OUT, "w") as f:
        f.write(summary + "\n")
    print(summary)


if __name__ == "__main__":
    main()
