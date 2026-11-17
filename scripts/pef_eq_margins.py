"""Equivalence (TOST on the HKSJ scale) of the L1 pooled estimate at several margins, from pef_analysis.py's pooled.json.

python3 -I scripts/pef_eq_margins.py [--run peakflow_paper/run_raw]
Margins: 2 pp (approved SESOI), 0.2 x median within-rider SD of evening %PB in pp (Paper 1's 0.2 SD), and the
same 0.2 SD in outcome-SD units. Also reports the smallest margin at which equivalence would hold.
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pef_analysis import tost  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser(); ap.add_argument("--run", default="peakflow_paper/run_raw"); a = ap.parse_args()
R = json.load(open(os.path.join(ROOT, a.run, "pooled.json")))
sd = R["desc"]["within_sd_pct_ev"]["median"]
print(f"median within-rider SD of evening %PB = {sd:.3f} pp; 0.2 SD = {0.2*sd:.3f} pp")
print("| Set | Scale | Margin | HKSJ 90% CI | TOST p | Equivalent? | Smallest margin that would pass |\n|---|---|---|---|---|---|---|")
for s, lab in [("all", "22-person"), ("slot20", "20-slot")]:
    for model, unit, margins in [("L1", "pp", [("±2 pp (approved)", 2.0), (f"±0.2 SD = ±{0.2*sd:.2f} pp", 0.2 * sd)]),
                                 ("L1_sdunits", "outcome SD", [("±0.2 SD", 0.2)])]:
        r = R["models"][model][f"{s}:PM_sd:z"]
        mn = max(abs(r["lo90_hk"]), abs(r["hi90_hk"]))
        for ml, m in margins:
            e = tost(r, m)
            print(f"| {lab} | {unit} | {ml} | [{e['lo90']:+.2f}, {e['hi90']:+.2f}] | {e['p_tost']:.3f} | {'yes' if e['equivalent'] else 'no'} | {mn:.2f} {unit} |")
