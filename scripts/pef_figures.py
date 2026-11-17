"""Paper 2 figures (S4) as standalone SVG, numpy/pandas only.

python3 -I scripts/pef_figures.py [--run peakflow_paper/run_raw] [--out peakflow_paper/figures]
Reads the outputs of pef_analysis.py and S2's calibration/tables/e1_summary.csv (cited, not recomputed).
"""
import argparse, json, os
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BLUE, ORANGE = "#2a78d6", "#eb6834"          # categorical slots 1-2: Bangkok, Chiang Mai
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#dcdbd6", "#fcfcfb"
CITY = {"BKK": BLUE, "CMI": ORANGE}
FONT = "font-family='Helvetica, Arial, sans-serif'"


class SVG:
    def __init__(self, w, h):
        self.w, self.h, self.e = w, h, []

    def add(self, s):
        self.e.append(s)

    def text(self, x, y, s, size=12, anchor="start", color=INK2, weight="normal"):
        s = str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        self.add(f"<text x='{x:.1f}' y='{y:.1f}' font-size='{size}' text-anchor='{anchor}' fill='{color}' font-weight='{weight}' {FONT}>{s}</text>")

    def line(self, x1, y1, x2, y2, color=GRID, w=1, dash=None):
        d = f" stroke-dasharray='{dash}'" if dash else ""
        self.add(f"<line x1='{x1:.1f}' y1='{y1:.1f}' x2='{x2:.1f}' y2='{y2:.1f}' stroke='{color}' stroke-width='{w}'{d}/>")

    def dot(self, x, y, color, r=4.5, fill=True, title=None):
        f = color if fill else SURF
        t = f"<title>{title}</title>" if title else ""
        self.add(f"<circle cx='{x:.1f}' cy='{y:.1f}' r='{r}' fill='{f}' stroke='{color}' stroke-width='2'>{t}</circle>")

    def diamond(self, x, lo, hi, y, color=INK, hh=7, title=None):
        t = f"<title>{title}</title>" if title else ""
        self.add(f"<polygon points='{lo:.1f},{y:.1f} {x:.1f},{y-hh:.1f} {hi:.1f},{y:.1f} {x:.1f},{y+hh:.1f}' fill='{color}'>{t}</polygon>")

    def save(self, path):
        with open(path, "w") as f:
            f.write(f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 {self.w} {self.h}' width='{self.w}' height='{self.h}'>"
                    f"<rect width='100%' height='100%' fill='{SURF}'/>" + "".join(self.e) + "</svg>\n")


def xaxis(s, x0, x1, lo, hi, ticks, ytop, ybot, label, fmt="{:+g}"):
    sx = lambda v: x0 + (v - lo) / (hi - lo) * (x1 - x0)
    for t in ticks:
        s.line(sx(t), ytop, sx(t), ybot, GRID)
        s.text(sx(t), ybot + 16, fmt.format(t), 11, "middle")
    s.text((x0 + x1) / 2, ybot + 34, label, 12, "middle", INK2)
    return sx


def legend_city(s, x, y):
    s.dot(x, y - 4, BLUE); s.text(x + 10, y, "Bangkok", 12, color=INK)
    s.dot(x + 90, y - 4, ORANGE); s.text(x + 100, y, "Chiang Mai", 12, color=INK)


def fig_forest(run, res, out):
    p = pd.read_csv(os.path.join(run, "per_person.csv"))
    p = p[p.model == "L1"].sort_values("b0").reset_index(drop=True)
    a, s20 = res["models"]["L1"]["all:PM_sd:z"], res["models"]["L1"]["slot20:PM_sd:z"]
    n = len(p); top, rh = 70, 20; bot = top + (n + 3) * rh
    S = SVG(680, bot + 70)
    S.text(20, 24, "Evening % of personal best per +1 within-rider SD of same-day PM2.5 (L1)", 14, color=INK, weight="bold")
    S.text(20, 44, "Per-rider OLS slope ± 95% CI (HC1); pooled = DL estimate with HKSJ 95% CI. Raw sensor PM. Dashed: ±2 pp SESOI.", 11)
    legend_city(S, 470, 62)
    lo = min(-8, float(np.floor((p.b0 - 1.96 * p.se0).min())))
    hi = max(6, float(np.ceil((p.b0 + 1.96 * p.se0).max())))
    x0, x1 = 185, 515
    sx = xaxis(S, x0, x1, lo, hi, [v for v in range(int(lo), int(hi) + 1) if v % 2 == 0], top - 6, bot, "pp of %PB per +1 SD PM")
    S.line(sx(0), top - 6, sx(0), bot, INK2, 1.2)
    for v in (-2, 2):
        S.line(sx(v), top - 6, sx(v), bot, INK2, 1, "4,3")
    for i, r in p.iterrows():
        y = top + i * rh + 6; c = CITY[r.city]
        S.line(sx(r.b0 - 1.96 * r.se0), y, sx(r.b0 + 1.96 * r.se0), y, c, 2)
        S.dot(sx(r.b0), y, c, title=f"{r.pid}: {r.b0:+.2f} pp (n {r.n})")
        S.text(x0 - 8, y + 4, f"{r.pid}" + ("" if r.slot20 else "*") + f"  n {int(r.n)}", 11, "end", INK)
    for j, (lab, r) in enumerate([("Pooled, all persons (k %d)" % a["k"], a), ("Pooled, 20-slot (k %d)" % s20["k"], s20)]):
        y = top + (n + 1 + j) * rh + 4
        S.diamond(sx(r["mu"]), sx(r["lo_hk"]), sx(r["hi_hk"]), y, title=f"{r['mu']:+.2f} [{r['lo_hk']:+.2f}, {r['hi_hk']:+.2f}]")
        S.text(x0 - 8, y + 4, lab, 11, "end", INK, "bold")
        S.text(x1 + 6, y + 4, f"{r['mu']:+.2f} [{r['lo_hk']:+.2f}, {r['hi_hk']:+.2f}] p {r['p_hk']:.2f}", 10, color=INK)
    S.text(20, bot + 58, "* not in the 20-slot set (reassigned Chiang Mai slot). R07 and R14 have < 12 complete evenings and are not shown.", 10)
    S.save(os.path.join(out, "fig1_forest_L1.svg"))


def fig_tertiles(res, out):
    S = SVG(680, 360); top, bot, x0, x1 = 80, 290, 90, 640
    S.text(20, 24, "Share of evening values below 80% of personal best, by the rider's own PM tertile (L2)", 14, color=INK, weight="bold")
    S.text(20, 44, "Mean of per-rider shares; 95% person-bootstrap CI (10,000). Tertiles are within rider, so no absolute PM is implied.", 11)
    sets = [("all", "22-person set", BLUE, True), ("slot20", "20-slot", BLUE, False)]
    hi = max(max(res["l2"][k][t + "_hi"] for t in ["low", "mid", "high"]) for k, *_ in sets)
    hi = float(np.ceil(hi / 5) * 5)
    sy = lambda v: bot - v / hi * (bot - top)
    for v in np.arange(0, hi + 0.1, 5):
        S.line(x0, sy(v), x1, sy(v), GRID); S.text(x0 - 8, sy(v) + 4, f"{v:g}%", 11, "end")
    cats = ["low", "mid", "high"]; cx = {c: x0 + (i + 0.5) * (x1 - x0) / 3 for i, c in enumerate(cats)}
    for j, (k, lab, col, fill) in enumerate(sets):
        r = res["l2"][k]; off = -14 if j == 0 else 14
        for c in cats:
            S.line(cx[c] + off, sy(r[c + "_lo"]), cx[c] + off, sy(r[c + "_hi"]), col, 2)
            S.dot(cx[c] + off, sy(r[c]), col, 5, fill, title=f"{lab} {c}: {r[c]:.1f}% [{r[c+'_lo']:.1f}, {r[c+'_hi']:.1f}]")
    for c in cats:
        S.text(cx[c], bot + 18, f"{c} PM (own tertile)", 12, "middle", INK)
    a, b = res["l2"]["all"], res["l2"]["slot20"]
    S.dot(x0 + 6, 62, BLUE, 5, True); S.text(x0 + 16, 66, f"22-person set (k {a['k']})", 12, color=INK)
    S.dot(x0 + 186, 62, BLUE, 5, False); S.text(x0 + 196, 66, f"20-slot (k {b['k']})", 12, color=INK)
    S.text(20, 330, f"High − low: {a['diff']:+.1f} pp [{a['diff_lo']:+.1f}, {a['diff_hi']:+.1f}] (22-person); "
                    f"{b['diff']:+.1f} pp [{b['diff_lo']:+.1f}, {b['diff_hi']:+.1f}] (20-slot).", 11, color=INK)
    S.save(os.path.join(out, "fig2_share_below80_by_tertile.svg"))


def fig_change(run, res, out):
    pr = pd.read_csv(os.path.join(run, "l3_pairs.csv"))
    g = pr.groupby("pid").agg(m=("delta", "mean"), sd=("delta", "std"), n=("delta", "size"), city=("city_c", "first"), slot20=("slot20", "first"))
    g["ci"] = 1.96 * g.sd / np.sqrt(g.n); g = g.sort_values("m").reset_index()
    n = len(g); top, rh = 70, 19; bot = top + n * rh
    S = SVG(680, bot + 90)
    S.text(20, 24, "Evening − morning % of personal best, same workday, per rider", 14, color=INK, weight="bold")
    S.text(20, 44, "Mean change ± 95% CI over the rider's paired workdays. Morning is not a pre-exposure baseline (sensor on at home first).", 11)
    legend_city(S, 470, 62)
    lo = float(np.floor((g.m - g.ci).min())) - 1; hi = float(np.ceil((g.m + g.ci).max())) + 1
    x0, x1 = 120, 650
    sx = xaxis(S, x0, x1, lo, hi, [v for v in range(int(lo), int(hi) + 1) if v % 2 == 0], top - 6, bot, "evening − morning, pp of %PB")
    S.line(sx(0), top - 6, sx(0), bot, INK2, 1.2)
    for i, r in g.iterrows():
        y = top + i * rh + 6; c = CITY[r.city]
        S.line(sx(r.m - r.ci), y, sx(r.m + r.ci), y, c, 2)
        S.dot(sx(r.m), y, c, title=f"{r.pid}: {r.m:+.2f} pp (n {r.n})")
        S.text(x0 - 8, y + 4, f"{r.pid}" + ("" if r.slot20 else "*") + f"  n {int(r.n)}", 11, "end", INK)
    l3 = res["models"]["L3"]["all:PM_between:z"]; d = res["desc"]["l3_pairs"]
    S.text(20, bot + 58, f"{d['n']} paired workdays; median change {d['median_delta']:+.1f} pp. L3 (change ~ z(PM between readings) + hours between): "
                         f"{l3['mu']:+.2f} pp per SD [{l3['lo_hk']:+.2f}, {l3['hi_hk']:+.2f}], k {l3['k']}.", 10)
    S.text(20, bot + 74, "* not in the 20-slot set.", 10)
    S.save(os.path.join(out, "fig3_evening_minus_morning.svg"))


def fig_gap(out):
    e = pd.read_csv(os.path.join(ROOT, "calibration/tables/e1_summary.csv"))
    e = e[e.analysis == "primary"]
    rows = []
    for city, groups in [("BKK", ["O01", "O02", "O03", "pooled"]), ("CMI", ["O04"])]:
        for gname in groups:
            for per in ["all", "Dec-Jan", "Feb-Apr"]:
                r = e[(e.city == city) & (e.group == gname) & (e.period == per)]
                if len(r) and pd.notna(r.ratio.iloc[0]):
                    rows.append(dict(city=city, group=gname, period=per, **r.iloc[0][["ratio", "ratio_ci_lo", "ratio_ci_hi", "share_higher", "n_hours"]].to_dict()))
    t = pd.DataFrame(rows)
    top, rh = 80, 20; n = len(t) + 2; bot = top + n * rh
    S = SVG(680, bot + 80)
    S.text(20, 24, "Rider-to-office PM2.5 ratio over matched city-hours (S2 E1, raw, provisional)", 14, color=INK, weight="bold")
    S.text(20, 44, "Ratio of means with S2's 95% CI; re-plotted from calibration/tables/e1_summary.csv, not recomputed. 1 = same air.", 11)
    legend_city(S, 470, 64)
    x0, x1, lo, hi = 230, 600, 0.8, 3.2
    sx = xaxis(S, x0, x1, lo, hi, [1, 1.5, 2, 2.5, 3], top - 6, bot, "rider ÷ office PM (ratio)", "{:g}")
    S.line(sx(1), top - 6, sx(1), bot, INK2, 1.2)
    y = top; prev = None
    for _, r in t.iterrows():
        if prev is not None and r.city != prev:
            y += rh
        prev = r.city
        c = CITY[r.city]; bold = r.group == "pooled" or r.city == "CMI"
        lab = f"{'Bangkok' if r.city == 'BKK' else 'Chiang Mai'} vs {'pooled office' if r.group == 'pooled' else r.group}, {r.period}"
        S.line(sx(r.ratio_ci_lo), y + 6, sx(r.ratio_ci_hi), y + 6, c, 2)
        S.dot(sx(r.ratio), y + 6, c, 4.5, r.period == "all", title=f"{lab}: {r.ratio:.2f} [{r.ratio_ci_lo:.2f}, {r.ratio_ci_hi:.2f}], {int(r.n_hours)} h")
        S.text(x0 - 8, y + 10, lab, 11, "end", INK, "bold" if (bold and r.period == "all") else "normal")
        S.text(x1 + 8, y + 10, f"{r.share_higher*100:.0f}% of h", 10)
        y += rh
    S.text(20, bot + 50, "Filled = whole study; hollow = season. Right column: share of matched hours with rider PM above office PM. O03 has no Dec-Jan hours.", 10)
    S.text(20, bot + 66, "Chiang Mai comparator is O04 only (O05/6006 dropped). Raw uncalibrated sensors; ratios assume similar sensor bias across nodes.", 10)
    S.save(os.path.join(out, "fig4_exposure_gap_S2.svg"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="peakflow_paper/run_raw")
    ap.add_argument("--out", default="peakflow_paper/figures")
    a = ap.parse_args()
    run = os.path.join(ROOT, a.run); out = os.path.join(ROOT, a.out); os.makedirs(out, exist_ok=True)
    res = json.load(open(os.path.join(run, "pooled.json")))
    fig_forest(run, res, out); fig_tertiles(res, out); fig_change(run, res, out); fig_gap(out)
    print("figures in", out)


if __name__ == "__main__":
    main()
