"""S3 figures (hand-written SVG; numpy/pandas only) from the outputs of scripts/pvt_analysis.py.
Usage: python3 -I scripts/pvt_figures.py [--run pvt_paper/run_raw] [--out pvt_paper/figures]
"""
import argparse, json, os
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser()
ap.add_argument("--run", default=os.path.join(ROOT, "pvt_paper", "run_raw"))
ap.add_argument("--out", default=os.path.join(ROOT, "pvt_paper", "figures"))
A = ap.parse_args(); os.makedirs(A.out, exist_ok=True)
J = json.load(open(os.path.join(A.run, "pooled.json"))); PO = J["pooled"]
PER = pd.read_csv(os.path.join(A.run, "per_rider_estimates.csv"))
C1, C2, INK, INK2, GRID, BG = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
FONT = 'font-family="Helvetica, Arial, sans-serif"'

def svg(w, h, body, title):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" {FONT}>'
            f'<title>{title}</title><rect width="{w}" height="{h}" fill="{BG}"/>' + "".join(body) + "</svg>")
def T(x, y, s, size=12, anchor="start", color=INK, weight="normal"):
    return f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" text-anchor="{anchor}" fill="{color}" font-weight="{weight}">{s}</text>'
def L(x1, y1, x2, y2, color=INK2, w=1, dash=None, op=1):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{color}" stroke-width="{w}" opacity="{op}"{d}/>'
def Cc(x, y, r=4.5, color=C1, tip=""):
    return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="{color}" stroke="{BG}" stroke-width="2"><title>{tip}</title></circle>'
def diamond(x, lo, hi, y, color=INK, tip=""):
    return f'<polygon points="{lo:.1f},{y:.1f} {x:.1f},{y-6:.1f} {hi:.1f},{y:.1f} {x:.1f},{y+6:.1f}" fill="{color}"><title>{tip}</title></polygon>'

def xaxis(x0, x1, y, vmin, vmax, ticks, label, ybottom_top=None):
    sx = lambda v: x0 + (v - vmin) / (vmax - vmin) * (x1 - x0)
    b = [L(x0, y, x1, y, INK2)]
    for t in ticks:
        b.append(L(sx(t), y, sx(t), y + 4, INK2)); b.append(T(sx(t), y + 17, f"{t:g}", 11, "middle", INK2))
        if ybottom_top: b.append(L(sx(t), ybottom_top[0], sx(t), ybottom_top[1], GRID))
    b.append(T((x0 + x1) / 2, y + 34, label, 12, "middle", INK2))
    return b, sx

def forest(rows, pooled, title, xlabel, fname, ticks, harm_note):
    """rows: list of (label, b, lo, hi). pooled: dict with mu, lo_hk, hi_hk."""
    w, top, rh = 640, 60, 20; h = top + rh * (len(rows) + 2) + 60
    vals = [r[2] for r in rows] + [r[3] for r in rows]
    vmin, vmax = min(ticks[0], np.nanmin(vals)), max(ticks[-1], np.nanmax(vals))
    x0, x1 = 110, w - 30; yb = top + rh * (len(rows) + 1.6)
    body = [T(16, 26, title, 15, weight="bold"), T(16, 44, harm_note, 11, color=INK2)]
    ax, sx = xaxis(x0, x1, yb, vmin, vmax, ticks, xlabel, (top - 6, yb)); body += ax
    body.append(L(sx(0), top - 6, sx(0), yb, INK2, 1))
    for i, (lab, b, lo, hi) in enumerate(rows):
        y = top + rh * i + 8
        body.append(T(x0 - 10, y + 4, lab, 11, "end", INK2))
        body.append(L(sx(max(lo, vmin)), y, sx(min(hi, vmax)), y, C1, 2))
        body.append(Cc(sx(b), y, 4, C1, f"{lab}: {b:+.2f} [{lo:+.2f}, {hi:+.2f}]"))
    y = top + rh * len(rows) + 14
    body.append(T(x0 - 10, y + 4, "Pooled (HKSJ)", 11, "end", INK, "bold"))
    body.append(diamond(sx(pooled["mu"]), sx(pooled["lo_hk"]), sx(pooled["hi_hk"]), y, INK,
                        f"pooled {pooled['mu']:+.2f} [{pooled['lo_hk']:+.2f}, {pooled['hi_hk']:+.2f}]"))
    open(os.path.join(A.out, fname), "w").write(svg(w, h, body, title))

def rider_rows(mid, f):
    p = PER[PER.model == mid].sort_values("pid")
    return [(r.pid, r["b_" + f], r["b_" + f] - 1.96 * r["se_" + f], r["b_" + f] + 1.96 * r["se_" + f]) for _, r in p.iterrows()]

# ---- Figure 2: forest per rider, PM z effect on Y1
forest(rider_rows("F1_Y1", "pm"), PO["F1_Y1"]["pm"],
       "Fig 2. Mean speed vs same-day PM2.5, per rider",
       "% change in mean speed per +1 within-rider SD of PM2.5", "fig2_forest_pm_y1.svg",
       [-6, -4, -2, 0, 2, 4, 6], "Per-rider OLS (95% CI); negative = slower. Raw sensor PM, within-rider z (calibration-invariant).")

# ---- Figure 4: moving vs not moving (prevalence + within-rider forest)
mv = pd.read_csv(os.path.join(A.run, "moving_by_rider.csv")).sort_values("pid")
rows = rider_rows("M1_Y1", "mv")
w, h = 900, 60 + 20 * 24 + 70; top = 60
body = [T(16, 26, "Fig 4. Tests taken while the sensor was moving at road speed (QC flag P8)", 15, weight="bold"),
        T(16, 44, "Left: share of GPS-evaluable tests per rider. Right: within-rider % change in mean speed, moving − not moving (riders with ≥ 3 of each).", 11, color=INK2)]
x0, x1 = 90, 380; yb = top + 20 * 23
ax, sx = xaxis(x0, x1, yb, 0, 0.7, [0, .1, .2, .3, .4, .5, .6, .7], "share of tests taken while moving", (top - 6, yb)); body += ax
for i, (_, r) in enumerate(mv.iterrows()):
    y = top + 20 * i + 8; s = r["sum"] / r["count"]
    body.append(T(x0 - 8, y + 4, r.pid, 11, "end", INK2))
    body.append(f'<rect x="{x0}" y="{y-6}" width="{max(sx(s)-x0,0):.1f}" height="12" rx="2" fill="{C1}"><title>{r.pid}: {int(r["sum"])}/{int(r["count"])}</title></rect>')
    body.append(T(sx(s) + 4, y + 4, f"{int(r['sum'])}/{int(r['count'])}", 10, color=INK2))
xs0, xs1 = 520, 870
ax, sx2 = xaxis(xs0, xs1, yb, -25, 15, [-25, -20, -15, -10, -5, 0, 5, 10, 15], "% change in mean speed, moving − not moving", (top - 6, yb)); body += ax
body.append(L(sx2(0), top - 6, sx2(0), yb, INK2))
rd = {r[0]: r for r in rows}
for i, (_, r) in enumerate(mv.iterrows()):
    y = top + 20 * i + 8
    if r.pid in rd:
        _, b, lo, hi = rd[r.pid]
        body.append(L(sx2(max(lo, -25)), y, sx2(min(hi, 15)), y, C2, 2)); body.append(Cc(sx2(max(min(b, 15), -25)), y, 4, C2, f"{r.pid}: {b:+.1f}% [{lo:+.1f}, {hi:+.1f}]"))
    else:
        body.append(T(sx2(0) + 6, y + 4, "too few moving tests", 10, color=INK2))
p = PO["M1_Y1"]["mv"]; y = top + 20 * 22 + 8
body.append(T(xs0 - 8, y + 4, "Pooled", 11, "end", INK, "bold"))
body.append(diamond(sx2(p["mu"]), sx2(p["lo_hk"]), sx2(p["hi_hk"]), y, INK, f"pooled {p['mu']:+.2f} [{p['lo_hk']:+.2f}, {p['hi_hk']:+.2f}]"))
open(os.path.join(A.out, "fig4_moving_tests.svg"), "w").write(svg(w, h, body, "Fig 4 moving tests"))

# ---- Figure 1: fatigue curve
cur = pd.read_csv(os.path.join(A.run, "fatigue_curve.csv")); rb = pd.read_csv(os.path.join(A.run, "fatigue_rider_bins.csv"))
labs = list(cur.hbin); w, h = 720, 440; x0, x1, y0, y1 = 80, 690, 70, 360
ymin, ymax = -15, 15
sy = lambda v: y1 - (v - ymin) / (ymax - ymin) * (y1 - y0)
sxb = lambda i: x0 + (i + 0.5) / len(labs) * (x1 - x0)
f2 = PO["F2a"]["hos"]
body = [T(16, 26, "Fig 1. Mean speed by hours on shift before the test", 15, weight="bold"),
        T(16, 44, f"Grey: each rider's bin means. Blue: mean of rider-bin means (95% CI across riders). Pooled F2a slope {f2['mu']:+.2f}%/h [{f2['lo_hk']:+.2f}, {f2['hi_hk']:+.2f}].", 11, color=INK2)]
for t in [-15, -10, -5, 0, 5, 10, 15]:
    body.append(L(x0, sy(t), x1, sy(t), GRID if t else INK2)); body.append(T(x0 - 8, sy(t) + 4, f"{t:+g}%" if t else "0", 11, "end", INK2))
body.append(T(22, (y0 + y1) / 2, "", 11))
body.append(f'<text transform="translate(24,{(y0+y1)/2}) rotate(-90)" font-size="12" text-anchor="middle" fill="{INK2}">% of own mean speed</text>')
for i, l in enumerate(labs): body.append(T(sxb(i), y1 + 18, l, 11, "middle", INK2))
body.append(T((x0 + x1) / 2, y1 + 38, "hours on shift before the test (GPS shift start)", 12, "middle", INK2))
for pid, g in rb.groupby("pid"):
    g = g.set_index("hbin").reindex(labs)
    pts = [(sxb(i), sy(np.clip(v, ymin, ymax))) for i, v in enumerate(g.Y1) if pd.notna(v)]
    if len(pts) > 1: body.append(f'<polyline points="{" ".join(f"{a:.1f},{b:.1f}" for a, b in pts)}" fill="none" stroke="{INK2}" stroke-width="1" opacity="0.25"><title>{pid}</title></polyline>')
pts = []
for i, r in cur.iterrows():
    body.append(L(sxb(i), sy(r.lo), sxb(i), sy(r.hi), C1, 2)); pts.append((sxb(i), sy(r["mean"])))
body.append(f'<polyline points="{" ".join(f"{a:.1f},{b:.1f}" for a, b in pts)}" fill="none" stroke="{C1}" stroke-width="2"/>')
for i, r in cur.iterrows(): body.append(Cc(sxb(i), sy(r["mean"]), 5, C1, f"{r.hbin}: {r['mean']:+.2f}% [{r.lo:+.2f}, {r.hi:+.2f}], {int(r['count'])} riders"))
open(os.path.join(A.out, "fig1_fatigue_curve.svg"), "w").write(svg(w, h, body, "Fig 1 fatigue curve"))

# ---- Figure 3: placebo before/after workday adjustment
pairs = [("PM after test (all), Y1", "F3a_Y1", "F3b_Y1", "pa"), ("PM after test, slowest 10% (Y3)", "F3a_Y3", "F3b_Y3", "pa"),
         ("PM after test, road part", "H2a_after_onshift", "H2b_after_onshift", "pa"), ("PM after test, home part", "H2a_after_home", "H2b_after_home", "pa"),
         ("Same-day PM before test, Y1 (reference)", "F3c", "F3d", "pm")]
w, h = 720, 70 + 46 * len(pairs) + 70; x0, x1 = 270, 690; top = 70; yb = top + 46 * len(pairs)
body = [T(16, 26, "Fig 3. Placebo: PM2.5 after the test, before and after workday adjustment", 15, weight="bold"),
        T(16, 44, "Blue: PM alone (+ time, clock hour). Orange: + hours on shift, km, clock hour, tests already today. HKSJ 95% CI; per +1 within-rider SD.", 11, color=INK2)]
ax, sx = xaxis(x0, x1, yb, -3, 1.5, [-3, -2, -1, 0, 1], "% change in speed per +1 SD of PM (Y1 or Y3)", (top - 10, yb)); body += ax
body.append(L(sx(0), top - 10, sx(0), yb, INK2))
for i, (lab, a, b, f) in enumerate(pairs):
    y = top + 46 * i + 12
    body.append(T(x0 - 10, y + 10, lab, 11, "end", INK2))
    for j, (mid, col) in enumerate([(a, C1), (b, C2)]):
        p = PO[mid][f]; yy = y + j * 14
        body.append(L(sx(max(p["lo_hk"], -3)), yy, sx(min(p["hi_hk"], 1.5)), yy, col, 2))
        body.append(Cc(sx(p["mu"]), yy, 4.5, col, f"{mid}: {p['mu']:+.2f} [{p['lo_hk']:+.2f}, {p['hi_hk']:+.2f}], p {p['p_hk']:.3f}"))
open(os.path.join(A.out, "fig3_placebo.svg"), "w").write(svg(w, h, body, "Fig 3 placebo"))

# ---- Figure 5: leave-one-rider-out ranges (small multiples, own axes)
loo = pd.read_csv(os.path.join(A.run, "loo.csv"))
panels = [("F1", "F1: speed per +1 SD PM (%)", PO["F1_Y1"]["pm"]["mu"]), ("F2a", "F2a: speed per hour on shift (%)", PO["F2a"]["hos"]["mu"]),
          ("F3a", "F3a: placebo, per +1 SD (%)", PO["F3a_Y1"]["pa"]["mu"]), ("shrink", "F3 shrinkage (share)", None),
          ("T1", "T1: slowest 10%, high vs low (%)", PO["T1_Y3"]["hi"]["mu"])]
w, ph = 720, 74; h = 60 + ph * len(panels) + 10
body = [T(16, 26, "Fig 5. Leave-one-rider-out: pooled estimate with each rider removed (22 refits)", 15, weight="bold"),
        T(16, 44, "Blue dots: refits. Black tick: full-sample estimate. Dashed: 0 (and the 50% shrinkage threshold).", 11, color=INK2)]
for i, (c, lab, full) in enumerate(panels):
    y = 60 + ph * i + 20; v = loo[c].to_numpy()
    if c == "shrink": full = None; lo_, hi_ = 0, 1
    else:
        lo_, hi_ = min(v.min(), 0, full), max(v.max(), 0, full); pad = (hi_ - lo_) * 0.15 + 1e-6; lo_ -= pad; hi_ += pad
    ticks = np.round(np.linspace(lo_, hi_, 5), 2)
    body.append(T(16, y - 4, lab, 12, color=INK))
    ax, sx = xaxis(200, 690, y + 14, lo_, hi_, ticks, ""); body += ax[:-1]
    ref = 0.5 if c == "shrink" else 0
    body.append(L(sx(ref), y - 10, sx(ref), y + 14, INK2, 1, "4,3"))
    for _, r in loo.iterrows(): body.append(Cc(sx(r[c]), y, 3.5, C1, f"without {r.pid}: {r[c]:+.3f}"))
    if full is not None: body.append(L(sx(full), y - 10, sx(full), y + 10, INK, 2.5))
open(os.path.join(A.out, "fig5_loo.svg"), "w").write(svg(w, h, body, "Fig 5 LOO"))
print("figures written to", A.out)
