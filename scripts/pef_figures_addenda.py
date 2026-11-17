"""Figures for Paper 2 addenda A (season) and B (night), SVG, numpy/pandas only.

python3 -I scripts/pef_figures_addenda.py
Reads peakflow_paper/run_season/, run_night/ and S2's calibration/tables/e3_monthly_pct_change.csv (cited).
"""
import json, os, sys
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pef_figures import SVG, BLUE, ORANGE, INK, INK2, GRID, CITY, xaxis, legend_city, ROOT

OUT = os.path.join(ROOT, "peakflow_paper/figures")
MONTHS = ["2026-01", "2026-02", "2026-03", "2026-04"]
MLAB = {"2026-01": "Jan", "2026-02": "Feb", "2026-03": "Mar", "2026-04": "Apr"}


def fig_season():
    m = pd.read_csv(os.path.join(ROOT, "peakflow_paper/run_season/season_months.csv"))
    r = json.load(open(os.path.join(ROOT, "peakflow_paper/run_season/season.json")))
    e3 = pd.read_csv(os.path.join(ROOT, "calibration/tables/e3_monthly_pct_change.csv")).set_index("month")
    S = SVG(680, 520)
    S.text(20, 24, "Evening % of personal best by month, per rider, in each city (Addendum A, exploratory)", 14, color=INK, weight="bold")
    S.text(20, 44, "Thin lines: riders (months with ≥ 2 values); thick: unweighted mean of riders. Lower strip: city PM % change vs Dec–Jan (S2 E3, raw).", 11)
    lo = 80; hi = 110
    for j, (city, title, ecol) in enumerate([("CMI", "Chiang Mai (air worsens Mar–Apr)", "cmi_pct_change_vs_decjan"),
                                              ("BKK", "Bangkok (air improves Mar–Apr)", "bkk_pct_change_vs_decjan")]):
        x0 = 60 + j * 320; x1 = x0 + 270; top, bot = 80, 330
        S.text(x0, 62, title, 12, color=CITY[city], weight="bold")
        sy = lambda v: bot - (v - lo) / (hi - lo) * (bot - top)
        for v in range(lo, hi + 1, 5):
            S.line(x0, sy(v), x1, sy(v), GRID); S.text(x0 - 6, sy(v) + 4, f"{v}", 10, "end")
        cx = {mm: x0 + (i + 0.5) * (x1 - x0) / 4 for i, mm in enumerate(MONTHS)}
        c = m[(m.city_c == city) & (m["size"] >= 2) & m.month.isin(MONTHS)]
        for pid, g in c.groupby("pid"):
            g = g.sort_values("month"); pts = " ".join(f"{cx[a]:.1f},{sy(min(max(b, lo), hi)):.1f}" for a, b in zip(g.month, g["mean"]))
            S.add(f"<polyline points='{pts}' fill='none' stroke='{CITY[city]}' stroke-opacity='0.35' stroke-width='1'><title>{pid}</title></polyline>")
        mean = c.groupby("month")["mean"].mean()
        pts = " ".join(f"{cx[a]:.1f},{sy(mean[a]):.1f}" for a in MONTHS if a in mean)
        S.add(f"<polyline points='{pts}' fill='none' stroke='{CITY[city]}' stroke-width='3'/>")
        for a in MONTHS:
            if a in mean:
                S.dot(cx[a], sy(mean[a]), CITY[city], 4.5, title=f"{city} {MLAB[a]}: {mean[a]:.1f}%")
            S.text(cx[a], bot + 16, MLAB[a], 11, "middle", INK)
        # PM strip (separate scale, separate panel: no second axis on the outcome)
        st, sb = 370, 450; plo, phi = -60, 220
        sp = lambda v: sb - (v - plo) / (phi - plo) * (sb - st)
        S.line(x0, sp(0), x1, sp(0), INK2, 1)
        for v in (-50, 0, 100, 200):
            S.text(x0 - 6, sp(v) + 4, f"{v:+d}%", 9, "end")
        for a in MONTHS:
            v = float(e3.loc[MLAB[a], ecol]); w = 30
            y0, y1 = sorted([sp(0), sp(v)])
            S.add(f"<rect x='{cx[a]-w/2:.1f}' y='{y0:.1f}' width='{w}' height='{max(y1-y0,1):.1f}' fill='{CITY[city]}' fill-opacity='0.6'><title>{city} {MLAB[a]} PM {v:+.1f}% vs Dec–Jan</title></rect>")
        S.text(x0, st - 8, "city PM, % change vs Dec–Jan", 10)
    t = r["T1_evening"]
    S.text(20, 490, f"Change haze − pre: Chiang Mai {t['cmi_dl']:+.1f} pp, Bangkok {t['bkk_dl']:+.1f} pp; CMI − BKK {t['diff_dl']:+.1f} pp "
                    f"[{t['diff_lo']:+.1f}, {t['diff_hi']:+.1f}], exact permutation p {t['p_perm']:.2f} (Holm {r['holm']['T1']:.2f}); 9 + 9 riders.", 10, color=INK)
    S.text(20, 506, "Raw sensor PM, relative change only. Months with < 2 values per rider omitted from the lines; the test uses windows, not months.", 10)
    S.save(os.path.join(OUT, "fig5_season_contrast.svg"))


def fig_night():
    p = pd.read_csv(os.path.join(ROOT, "peakflow_paper/run_night/per_person.csv"))
    r = json.load(open(os.path.join(ROOT, "peakflow_paper/run_night/night.json")))
    n1 = p[p.model == "N1"].set_index("pid"); n2 = p[p.model == "N2"].set_index("pid")
    order = n1.sort_values("b0").index.tolist()
    n = len(order); top, rh = 80, 22; bot = top + (n + 3) * rh
    S = SVG(680, bot + 70)
    S.text(20, 24, "Morning % of personal best per +1 within-rider SD of PM (Addendum B)", 14, color=INK, weight="bold")
    S.text(20, 44, "Filled: night air (00:00 → morning value). Hollow: placebo, air after the value (→ 24:00). ± 95% CI; dashed ± 2 pp.", 11)
    S.dot(380, 62, INK2, 4.5, True); S.text(390, 66, "night (N1)", 11, color=INK)
    S.dot(470, 62, INK2, 4.5, False); S.text(480, 66, "placebo (N2)", 11, color=INK)
    S.text(560, 66, "colour = city", 11)
    x0, x1, lo, hi = 200, 500, -8, 8
    sx = xaxis(S, x0, x1, lo, hi, list(range(-8, 9, 2)), top - 6, bot, "pp of %PB per +1 SD PM")
    S.line(sx(0), top - 6, sx(0), bot, INK2, 1.2)
    for v in (-2, 2):
        S.line(sx(v), top - 6, sx(v), bot, INK2, 1, "4,3")
    cl = lambda v: min(max(v, lo), hi)
    for i, pid in enumerate(order):
        y = top + i * rh + 6; c = CITY[n1.loc[pid, "city"]]
        for tab, off, fill in [(n1, -4, True), (n2, 4, False)]:
            b, se = tab.loc[pid, "b0"], tab.loc[pid, "se0"]
            S.line(sx(cl(b - 1.96 * se)), y + off, sx(cl(b + 1.96 * se)), y + off, c, 1.5)
            S.dot(sx(cl(b)), y + off, c, 3.5, fill, title=f"{pid} {'night' if fill else 'placebo'}: {b:+.2f}")
        S.text(x0 - 8, y + 4, f"{pid}" + ("" if n1.loc[pid, "slot20"] else "*") + f"  n {int(n1.loc[pid, 'n'])}", 11, "end", INK)
    for j, sset in enumerate(["all", "slot20"]):
        y = top + (n + 1 + j) * rh + 4
        a, b = r["models"]["N1"][f"{sset}:PM_night:z"], r["models"]["N2"][f"{sset}:PM_after24:z"]
        S.diamond(sx(a["mu"]), sx(a["lo_hk"]), sx(a["hi_hk"]), y - 4, INK, 5)
        S.add(f"<polygon points='{sx(b['lo_hk']):.1f},{y+5:.1f} {sx(b['mu']):.1f},{y:.1f} {sx(b['hi_hk']):.1f},{y+5:.1f} {sx(b['mu']):.1f},{y+10:.1f}' fill='none' stroke='{INK}' stroke-width='1.5'/>")
        S.text(x0 - 8, y + 4, f"Pooled, {'all persons' if sset == 'all' else '20-slot'} (k {int(a['k'])})", 11, "end", INK, "bold")
        S.text(x1 + 6, y, f"night {a['mu']:+.2f} [{a['lo_hk']:+.2f}, {a['hi_hk']:+.2f}]", 10, color=INK)
        S.text(x1 + 6, y + 12, f"placebo {b['mu']:+.2f} [{b['lo_hk']:+.2f}, {b['hi_hk']:+.2f}]", 10)
    S.text(20, bot + 50, f"Decision rule (fixed before the run): {r['rule']['verdict']}. 528 mornings; R14 has < 12 and is not shown. * not in the 20-slot set.", 10, color=INK)
    S.save(os.path.join(OUT, "fig6_night_vs_placebo.svg"))


if __name__ == "__main__":
    fig_season(); fig_night(); print("figures in", OUT)
