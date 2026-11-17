"""Render the result tables of pef_analysis.py (pooled.json) as Markdown. numpy/pandas only.

python3 -I scripts/pef_report.py [--run peakflow_paper/run_raw] > tables.md
"""
import argparse, json, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def f(v, d=2, sign=True):
    return "—" if v is None else (f"{v:+.{d}f}" if sign else f"{v:.{d}f}")


def row(label, r, scale_note=""):
    if not r:
        return f"| {label} | — | — | — | — | — | — | — |"
    return (f"| {label} | {int(r['k'])} | {int(r['n'])} | {f(r['mu'])} | [{f(r['lo_hk'])}, {f(r['hi_hk'])}] | "
            f"{r['p_hk']:.3f} | {r['p_dl']:.3f} | {r['i2']:.0f}% / {r['share_neg']*100:.0f}% |")


HDR = "| Model | k | n | β | HKSJ 95% CI | p HKSJ | p DL | I² / share β<0 |\n|---|---|---|---|---|---|---|---|"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--run", default="peakflow_paper/run_raw"); a = ap.parse_args()
    R = json.load(open(os.path.join(ROOT, a.run, "pooled.json")))
    M = R["models"]; o = []
    P = lambda name, key: M.get(name, {}).get(key)

    def pair(title, name, key, unit=""):
        o.append(row(f"{title}{unit}", P(name, "all:" + key)))
        o.append(row(f"  20-slot", P(name, "slot20:" + key)))

    o.append("### Table A. L1 family (pp of %PB per unit shown; evening values)\n" + HDR)
    pair("**L1 primary** (per +1 SD)", "L1", "PM_sd:z")
    pair("L1 per within-rider IQR", "L1_iqr", "PM_sd:iqr")
    pair("L1 per +10 raw units `[CAL: raw]`", "L1_raw10", "PM_sd:raw10")
    pair("L1-cum (z cumulative)", "L1_cum", "PM_cum:z")
    pair("L1-T (+ days since enrolment)", "L1_T", "PM_sd:z")
    pair("L1 in outcome-SD units", "L1_sdunits", "PM_sd:z")
    o.append("")
    o.append("### Table B. Equivalence (TOST on the HKSJ 90% CI)\n| Set | Margin | HKSJ 90% CI | TOST p | Equivalent? |\n|---|---|---|---|---|")
    for s, lab in [("all", "22-person"), ("slot20", "20-slot")]:
        for u, ul in [("pp", "±2 pp (SESOI, approved)"), ("sd", "±0.2 outcome SD (Paper 1, secondary)")]:
            e = R["eq"][s][u]
            o.append(f"| {lab} | {ul} | [{f(e['lo90'], 3)}, {f(e['hi90'], 3)}] | {e['p_tost']:.3f} | {'yes' if e['equivalent'] else 'no'} |")
    o.append("")
    o.append("### Table C. L2 share of evening values < 80% PB by own PM tertile (mean of per-person shares, %; 95% person bootstrap)\n"
             "| Set | k | n | Low | Mid | High | High − low (pp) |\n|---|---|---|---|---|---|---|")
    for s, lab in [("all", "22-person"), ("slot20", "20-slot")]:
        r = R["l2"][s]
        o.append(f"| {lab} | {r['k']} | {r['n']} | {r['low']:.1f} [{r['low_lo']:.1f}, {r['low_hi']:.1f}] | {r['mid']:.1f} [{r['mid_lo']:.1f}, {r['mid_hi']:.1f}] | "
                 f"{r['high']:.1f} [{r['high_lo']:.1f}, {r['high_hi']:.1f}] | {f(r['diff'], 1)} [{f(r['diff_lo'], 1)}, {f(r['diff_hi'], 1)}] |")
    o.append("")
    o.append("### Table D. L3 within-day change, L4 work terms\n" + HDR)
    pair("**L3** Δ(evening − morning) per +1 SD PM between readings", "L3", "PM_between:z")
    pair("L4a GPS hours on shift (per h)", "L4a", "hours_on_shift_before_test:none")
    pair("L4b km ridden (per 10 km)", "L4b", "km10:none")
    pair("L4c hours on shift, with z(PM)", "L4c", "hours_on_shift_before_test:none")
    pair("L4c z(PM), with hours", "L4c", "PM_sd:z")
    pair("L4d km, with z(PM)", "L4d", "km10:none")
    pair("L4d z(PM), with km", "L4d", "PM_sd:z")
    pair("L4b on the L4d sample (no PM)", "L4b_samesample", "km10:none")
    o.append("")
    o.append("### Table E. Sensitivity of L1 (per +1 SD)\n" + HDR)
    pair("L5a last entered value per period", "L5a", "PM_sd:z")
    pair("L5b + F4 indicator", "L5b", "PM_sd:z")
    if "L1_pbstable" in M:
        pair("PB-stability refit (exploratory)", "L1_pbstable", "PM_sd:z")
    pair("S-med (median)", "S_med", "PM_med:z")
    pair("S-clip (clipped mean)", "S_clip", "PM_clip:z")
    pair("S-onshift (GPS shift start → reading)", "S_onshift", "PM_on:z")
    pair("S-cov (no coverage rule)", "S_cov", "PM_sd_nocov:z")
    pair("S-month (+ calendar-month dummies)", "S_month", "PM_sd:z")
    lp = R["lopo"]
    o.append(f"\nL5d leave-one-person-out (k {lp['k']} refits): β {f(lp['mu_min'])} to {f(lp['mu_max'])}; HKSJ p {lp['p_min']:.3f} (without {lp['drop_min_p']}) "
             f"to {lp['p_max']:.3f} (without {lp['drop_max_p']}); refits whose HKSJ CI excludes 0: {lp['n_ci_excl0']}.")
    ps = R["pb_stability"]
    o.append(f"PB stability (second-half PB ÷ first-half PB): median {ps['ratio_median']:.3f}, range {ps['ratio_min']:.3f}–{ps['ratio_max']:.3f}; "
             f"> 10% apart: {', '.join(ps['unstable']) or 'none'}; R12 {ps['r12']:.3f}, R14 {ps['r14']:.3f}.\n")
    o.append("### Table F. City (L6) and placebo exposures (L7)\n" + HDR)
    o.append(row("L1 Bangkok", P("L1", "BKK:PM_sd:z")))
    o.append(row("L1 Chiang Mai", P("L1", "CMI:PM_sd:z")))
    o.append(row("L1 Chiang Mai, 20-slot", P("L1_CMI_slot20", "all:PM_sd:z")))
    l6 = R["l6"]
    o.append(f"| CMI − BKK difference (DL; t, {l6['df']:.0f} df) | | | {f(l6['diff'])} | [{f(l6['lo'])}, {f(l6['hi'])}] | {l6['p']:.3f} | | |")
    pair("L7a z(PM after the reading)", "L7a", "PM_after:z")
    pair("L7b z(PM two days later)", "L7b", "PM_lead2:z")
    pair("L7c joint: same-day term", "L7c", "PM_sd:z")
    pair("L7c joint: after term", "L7c", "PM_after:z")
    if R.get("l7b_note"):
        o.append(f"\nL7b: {R['l7b_note']}")
    o.append("")
    b2, b3 = R["b2"], R["b3"]
    o.append("### Table G. Bangkok pilot (n = 10; validity and description only)\n| Item | Estimate | 95% CI |\n|---|---|---|")
    m = b2["mf_ratio"]
    o.append(f"| B2 PB men ÷ women ({m['n_m']} M / {m['n_f']} F) | {m['ratio']:.2f} | [{m['lo']:.2f}, {m['hi']:.2f}] |")
    g = b2["age"]
    o.append(f"| B2 PB vs age, OLS slope `[L/min per 10 y]` | {f(g['slope_per10y'], 1)} | [{f(g['lo'], 1)}, {f(g['hi'], 1)}] |")
    o.append(f"| B2 PB vs age, Spearman ρ | {f(g['spearman'])} | [{f(g['rho_lo'])}, {f(g['rho_hi'])}] |")
    for k, lab in [("age", "B3 L1 slope on age (pp per SD, per 10 y)"), ("sex_m_minus_f", "B3 L1 slope, men − women")]:
        r = b3[k]
        o.append(f"| {lab} (k {r['k']}) | {f(r['coef'])} | [{f(r['lo'])}, {f(r['hi'])}], p {r['p']:.3f} |")
    o.append(f"\nChiang Mai repeat rule fired: **{'yes' if b3['cmi_repeat_rule_fired'] else 'no'}**.")
    n_est = sum(1 for mm in M.values() for v in mm.values() if isinstance(v, dict) and "mu" in v)
    o.append(f"\nPooled estimates in this run: {n_est} (model × person-set × term), plus L2 2, L6 1, LOPO {lp['k']}, B2 3, B3 2.")
    print("\n".join(o))


if __name__ == "__main__":
    main()
