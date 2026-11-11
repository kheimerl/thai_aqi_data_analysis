"""S2: relative exposure and city comparison on raw sensor PM2.5 (calibration/RELATIVE_SPEC.md).

Run:      python3 -I scripts/calib_run_all.py                      (raw, default)
Nov 28:   python3 -I scripts/calib_run_all.py --cal-json calibration/ait_equation.json --out-dir calibration/calibrated
          (json: {"a": .., "b": .., "c_rh": 0}; calibrated = a + b*pm + c_rh*rh, applied per minute and per hour).
          --pm-col pm_calibrated uses S1's hourly column for E1/E3 only; E2 (minute data) then stays RAW and is
          labelled raw, because the minute files have no calibrated column (audit SF3). Prefer --cal-json.
Options:  --include-6006 keeps node 6006 (O05, probable) in the office arm (run-1 setting; spec §7).
Outputs go to calibration/ (or --out-dir). numpy/pandas only. No PVT / peak-flow data used.

v3 (2026-11-11): audit fixes SF2-SF8 (audits/s2_calibration_2026-11-10.md) and the 20-slot sensitivity
(spec §9: riders without R12, R14). See RELATIVE_SPEC.md §7-§9.
"""
import argparse, json, os, zlib
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FND = os.path.join(ROOT, "foundation")
SEED = 20261110
NBOOT = 2000
MIN_CLUSTERS_CI = 5          # no bootstrap CI with fewer clusters (spec §8)
# Office table owned by this script (audit SF8): independent of how nodes.csv labels 6006.
OFFICE = {6009: "O01", 6010: "O02", 6033: "O03", 6007: "O04", 6006: "O05"}
SLOT20_EXCLUDE = {"R12", "R14"}   # replacement riders on 6021 (from Mar 4) and 6022 (from Mar 27)

ap = argparse.ArgumentParser()
ap.add_argument("--pm-col", default="pm_mean")
ap.add_argument("--cal-json", default=None)
ap.add_argument("--out-dir", default=os.path.join(ROOT, "calibration"))
ap.add_argument("--include-6006", action="store_true",
                help="keep node 6006 (O05, probable) in the office arm (run-1 setting). Default since "
                     "RELATIVE_SPEC.md §7: 6006 is ignored in E1/E2.")
args = ap.parse_args()
OUT = args.out_dir
for sub in ("tables", "figures"):
    os.makedirs(os.path.join(OUT, sub), exist_ok=True)
CAL = json.load(open(args.cal_json)) if args.cal_json else None
LABEL = "raw" if (CAL is None and args.pm_col == "pm_mean") else "calibrated"
LABEL_E2 = "raw" if CAL is None else "calibrated"          # SF3: minute data need --cal-json
if LABEL != LABEL_E2:
    print(f"NOTE: E1/E3 use {args.pm_col} ({LABEL}); E2 stays raw (minute files have no calibrated column).")


def cal(pm, rh):
    if CAL is None:
        return pm
    c = CAL.get("c_rh", 0.0)
    out = CAL["a"] + CAL["b"] * pm
    return out + c * rh if c != 0 else out      # no RH term: hours with missing RH are kept


CALP = []  # (where, quantity, value)


def calp(where, what, v):
    CALP.append((where, what, v))
    return f"[CAL: {v:.1f}]"


def boot_idx(n, key):
    """SF2: one generator per estimate, seeded by (SEED, crc32(key)), so CIs do not depend on run order.
    The key names the estimate (group/city/period), not the analysis variant, so the primary, 20-slot and
    sensitivity rows of one estimate share the same draws (common random numbers): identical data give identical CIs."""
    g = np.random.default_rng([SEED, zlib.crc32(key.encode("utf-8"))])
    return g.integers(0, n, size=(NBOOT, n))


def ci(a):
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    if len(a) == 0:
        return (np.nan, np.nan)
    return (np.percentile(a, 2.5), np.percentile(a, 97.5))


def fmt(x, d=2):
    return "n/a" if x is None or not np.isfinite(x) else f"{x:.{d}f}"


def pct(x):
    return "n/a" if x is None or not np.isfinite(x) else f"{100 * x:.0f}%"


def spct(x):
    if not np.isfinite(x):
        return "n/a"
    r = int(round(x))
    return f"{r:+d}%" if r != 0 else "0%"


RH_NOTE = "S-RH (uninformative: in-housing RH, 1 valid hour > 85; CLAUDE.md rule 7)"

# ---------------------------------------------------------------- load hourly
nodes = pd.read_csv(os.path.join(FND, "nodes.csv"))
CLS = {n: ("office" if n in OFFICE else c) for n, c in zip(nodes.node, nodes["class"])}
home = dict(zip(nodes.node, nodes.city))
H = pd.read_csv(os.path.join(FND, "pm_hourly_by_node.csv"))
H = H[H.node != 6008].copy()
H["cls"] = H.node.map(CLS)
H.loc[H.node.isin(OFFICE), "person"] = H.loc[H.node.isin(OFFICE), "node"].map(OFFICE)
H = H[H.cls.isin(["rider", "office"]) & H.person.notna()]
H["pm"] = cal(H["pm_mean"], H["rh_mean"]) if CAL is not None else H[args.pm_col]
if H["pm"].notna().sum() == 0:
    raise SystemExit(f"--pm-col {args.pm_col} is empty in pm_hourly_by_node.csv; fill it or use --cal-json.")
H_assigned = H[["node", "hour_unix", "person"]].copy()  # all assigned hours (for E2 shift detection)
raw_zero = H["pm_mean"] == 0
H = H[(H.minutes_pm >= 10) & H.pm_mean.between(0, 500) & ~raw_zero & H.pm.notna()].copy()
H["city"] = np.where(H.lat_median.notna(), np.where(H.lat_median > 17, "CMI", "BKK"), H.node.map(home))
H["hour_local"] = pd.to_datetime(H.hour_local)
H["date"] = H.hour_local.dt.strftime("%Y-%m-%d")
H["month"] = H.hour_local.dt.month
H["clock"] = H.hour_local.dt.hour
H["period"] = np.where(H.month.isin([12, 1]), "Dec-Jan", "Feb-Apr")
H["rhflag"] = H.rh_mean > 85
H["slot20"] = ~H.person.isin(SLOT20_EXCLUDE)
MONTHS = {12: "Dec", 1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr"}

# --------------------------------------------- node x date met summary (for AIT)
Hall = pd.read_csv(os.path.join(FND, "pm_hourly_by_node.csv"))
Hall["date"] = Hall.hour_local.str[:10]
met = Hall[Hall.node != 6008].groupby(["node", "date"]).agg(
    temp_mean=("temp_mean", "mean"), temp_min=("temp_mean", "min"), temp_max=("temp_mean", "max"),
    rh_mean=("rh_mean", "mean"), rh_min=("rh_mean", "min"), rh_max=("rh_mean", "max"),
    hours_rh_gt85=("rh_mean", lambda s: int((s > 85).sum())),
    pm_hours=("minutes_pm", lambda s: int((s >= 10).sum()))).reset_index()
met["class"] = met.node.map(CLS)
met.round(2).to_csv(os.path.join(OUT, "node_daily_met.csv"), index=False)

R = H[H.cls == "rider"]
R20 = R[R.slot20]
O_all = H[H.cls == "office"]
O = O_all if args.include_6006 else O_all[O_all.node != 6006]

# rider node medians (SF7: absolute values only via CAL_PENDING)
_med = R.groupby("node").pm.median()
calp("Spec §6 / STATUS", "lowest rider-node median (sensor-to-sensor spread)", _med.min())
calp("Spec §6 / STATUS", "highest rider-node median (sensor-to-sensor spread)", _med.max())

# ------------------------------------------------- minute files (E2 and N1)
MIN = {}
for nd in sorted(set(R.node) | set(O_all.node)):
    M = pd.read_csv(os.path.join(FND, "_work", f"minute_{nd}.csv"))
    M["hour_unix"] = (M.minute_unix // 3600) * 3600
    MIN[nd] = M


# ======================================================================== E1
def e1_match(Rsub, Osub, min_riders=2):
    g = Rsub.groupby(["city", "hour_unix"]).agg(rider=("pm", "mean"), n_r=("pm", "size"),
                                                 r_rhfrac=("rhflag", "mean")).reset_index()
    m = Osub.merge(g, on=["city", "hour_unix"], how="inner")
    return m[m.n_r >= min_riders].copy()


def e1_stats(m, key, offset=0.0, do_boot=True):
    if len(m) == 0:
        return dict(n_hours=0, n_dates=0)
    r = m.rider.values + offset
    o = m.pm.values + offset
    out = dict(n_hours=len(m), n_dates=m.date.nunique(), med_riders=float(np.median(m.n_r)),
               ratio=r.sum() / o.sum(), share_higher=float(np.mean(r > o)),
               med_hourly_ratio=float(np.median(r[o >= 1] / o[o >= 1])) if (o >= 1).any() else np.nan,
               office_mean=o.mean(), rider_mean=r.mean(), diff=(r - o).mean())
    if do_boot and m.date.nunique() >= MIN_CLUSTERS_CI:
        d = pd.DataFrame({"date": m.date.values, "r": r, "o": o, "h": (r > o).astype(float), "n": 1.0})
        D = d.groupby("date")[["r", "o", "h", "n"]].sum().values
        S = D[boot_idx(len(D), "E1|" + key)].sum(axis=1)
        out["ratio_ci"] = ci(S[:, 0] / S[:, 1])
        out["share_ci"] = ci(S[:, 2] / S[:, 3])
    return out


m_all = e1_match(R, O)
m_20 = e1_match(R20, O)
mh = m_all.merge(m_20[["node", "hour_unix", "rider", "n_r"]].rename(columns={"rider": "rider20", "n_r": "n_r20"}),
                 on=["node", "hour_unix"], how="left")
mh = mh.rename(columns={"pm": f"office_{LABEL}", "rider": f"rider_mean_{LABEL}", "node": "office_node",
                        "person": "office", "n_r": "n_riders", "rider20": f"rider_mean_20slot_{LABEL}",
                        "n_r20": "n_riders_20slot", "r_rhfrac": "rider_rhflag_share"})
mh[["office", "office_node", "city", "hour_local", "date", "clock", "period", f"office_{LABEL}", f"rider_mean_{LABEL}",
    "n_riders", f"rider_mean_20slot_{LABEL}", "n_riders_20slot", "rhflag", "rider_rhflag_share"]] \
    .round(3).to_csv(os.path.join(OUT, "tables", "e1_matched_hours.csv"), index=False)

e1_rows = []


def add_e1(group, city, period, m, analysis="primary", offset=0.0, key_group=None):
    s = e1_stats(m, f"{key_group or group}|{city}|{period}", offset)
    s.update(group=group, city=city, period=period, analysis=analysis)
    e1_rows.append(s)
    return s


def sel(m, city, who, period):
    mc = m[m.city == city]
    if who == "O04+trip":
        pass
    elif who != "pooled":
        mc = mc[mc.person == who]
    if period in ("Dec-Jan", "Feb-Apr"):
        mc = mc[mc.period == period]
    elif period in MONTHS.values():
        mc = mc[mc.month == {v: k for k, v in MONTHS.items()}[period]]
    return mc


for city in ["BKK", "CMI"]:
    for who in sorted(O.person.unique()) + ["pooled"]:
        if len(sel(m_all, city, who, "all")) == 0:
            continue
        periods = ["all", "Dec-Jan", "Feb-Apr"]
        if who == "pooled":
            periods += [MONTHS[mo] for mo in [12, 1, 2, 3, 4]
                        if sel(m_all, city, who, MONTHS[mo]).date.nunique() >= 5]
        for per in periods:
            add_e1(who, city, per, sel(m_all, city, who, per))
            add_e1(who, city, per, sel(m_20, city, who, per), "20-slot")
    mc = sel(m_all, city, "pooled", "all")
    add_e1("pooled", city, "all", mc[~mc.rhflag & (mc.r_rhfrac <= 0.5)], RH_NOTE)
    add_e1("pooled", city, "all", mc, "S-int -5", -5.0)
    add_e1("pooled", city, "all", mc, "S-int +5", +5.0)
    add_e1("pooled", city, "all", mc[mc.clock.between(7, 19)], "S-daytime")
    if city == "CMI":
        add_e1("pooled", city, "all", mc[mc.person == "O04"], "S-O04-only (no O01 CMI-trip hours; SF1)", key_group="O04")
        add_e1("pooled", city, "all", sel(m_20, city, "O04", "all"), "S-O04-only, 20-slot", key_group="O04")
        if args.include_6006:
            add_e1("pooled", city, "all", mc[mc.person != "O05"], "S-without-6006 (O04 + O01 CMI-trip hours)")
        else:
            m6 = e1_match(R, O_all)
            add_e1("pooled", city, "all", m6[m6.city == city], "S-incl-6006 (O04 + O05 + O01 CMI-trip hours; run-1 primary)")
    lo = []
    for nd in sorted(R[R.city == city].node.unique()):
        mm = e1_match(R[R.node != nd], O)
        lo.append(e1_stats(mm[mm.city == city], "loro", do_boot=False)["ratio"])
    e1_rows.append(dict(group="pooled", city=city, period="all", analysis="S-LORO",
                        ratio=np.nan, loro_min=min(lo), loro_max=max(lo)))

# N1 (exploratory, suggested by S7): hourly means rebuilt without exact-zero minutes
NZ_OK = not (CAL is None and args.pm_col != "pm_mean")
if NZ_OK:
    nz = []
    for nd, M in MIN.items():
        x = M[(M.n_pm > 0) & M.pm.between(0, 500) & (M.pm != 0)].copy()
        x["w_pm"] = x.pm * x.n_pm
        x["w_rh"] = x.rh * x.n_pm
        g = x.groupby("hour_unix").agg(s_pm=("w_pm", "sum"), s_rh=("w_rh", "sum"), n=("n_pm", "sum"),
                                       nmin=("pm", "size")).reset_index()
        g["node"] = nd
        g["pm_nz"] = cal(g.s_pm / g.n, g.s_rh / g.n)
        nz.append(g[g.nmin >= 10][["node", "hour_unix", "pm_nz"]])
    NZ = pd.concat(nz)
    Hn = H.merge(NZ, on=["node", "hour_unix"], how="inner").drop(columns="pm").rename(columns={"pm_nz": "pm"})
    Rn, On = Hn[Hn.cls == "rider"], Hn[(Hn.cls == "office") & (args.include_6006 | (Hn.node != 6006))]
    mn = e1_match(Rn, On)
    for city in ["BKK", "CMI"]:
        add_e1("pooled", city, "all", mn[mn.city == city], "S-zero-min (exploratory; audit N1)")

E1 = pd.DataFrame(e1_rows)
ABS1 = {"office_mean": f"office_mean_{LABEL}", "rider_mean": f"rider_mean_{LABEL}", "diff": f"rider_minus_office_{LABEL}"}
E1c = E1.copy().rename(columns=ABS1)
for k in ["ratio_ci", "share_ci"]:
    E1c[k + "_lo"] = E1c[k].map(lambda t: t[0] if isinstance(t, tuple) else np.nan)
    E1c[k + "_hi"] = E1c[k].map(lambda t: t[1] if isinstance(t, tuple) else np.nan)
    E1c = E1c.drop(columns=k)
E1c.round(4).to_csv(os.path.join(OUT, "tables", "e1_summary.csv"), index=False)

# ======================================================================== E2
hr_key = H_assigned.merge(H[["node", "hour_unix", "rhflag", "city"]], on=["node", "hour_unix"], how="left")
pdays = []
for nd in sorted(set(R.node) | set(O.node)):
    M = MIN[nd].merge(hr_key[hr_key.node == nd].drop(columns="node"), on="hour_unix", how="left")
    valid_hours = set(H[H.node == nd].hour_unix)
    M["valid_pm"] = M.hour_unix.isin(valid_hours) & M.pm.between(0, 500) & (M.n_pm > 0)
    M["pmc"] = cal(M.pm, M.rh)
    M["wd"] = pd.to_datetime(M.minute_unix + 7 * 3600 - 4 * 3600, unit="s").dt.strftime("%Y-%m-%d")
    M = M[M.person.notna()]  # keep only minutes inside the person's assigned hours
    for wd, g in M.groupby("wd"):
        g = g.sort_values("minute_unix")
        person = g.person.dropna()
        if person.empty:
            continue
        person = person.mode().iloc[0]
        nvalid_h = g.loc[g.valid_pm, "hour_unix"].nunique()
        n_speed = int(g.speed_kmh.notna().sum())       # "GPS minutes" = minutes with a speed value (spec §8)
        if nvalid_h < 12 or n_speed < 30:
            continue
        t0, t1 = int(g.minute_unix.min()), int(g.minute_unix.max())
        grid = np.arange(t0, t1 + 60, 60)
        mv = np.zeros(len(grid))
        pos = ((g.minute_unix.values - t0) // 60).astype(int)
        mv[pos] = (g.speed_kmh.values > 5).astype(float)
        cs = np.concatenate([[0], np.cumsum(mv)])
        n = len(grid)
        fwd = cs[np.minimum(np.arange(n) + 15, n)] - cs[np.arange(n)]
        bwd = cs[np.arange(n) + 1] - cs[np.maximum(np.arange(n) - 14, 0)]
        if not (fwd >= 8).any():
            continue
        s_start = grid[np.argmax(fwd >= 8)]
        s_end = grid[n - 1 - np.argmax((bwd >= 8)[::-1])]
        if s_end - s_start < 3600:
            continue
        v = g[g.valid_pm]
        on = (v.minute_unix >= s_start) & (v.minute_unix <= s_end)
        if on.sum() < 30 or (~on).sum() < 30:            # spec §8 implementation rule
            continue
        p = v.pmc.values
        sp = v.speed_kmh.values
        day_city = v.city.mode().iloc[0] if v.city.notna().any() else home[nd]   # N3: GPS city of the day
        rec = dict(person=person, node=nd, cls=CLS[nd], city=day_city, workday=wd,
                   slot20=person not in SLOT20_EXCLUDE,
                   shift_start=pd.to_datetime(s_start + 7 * 3600, unit="s").strftime("%H:%M"),
                   shift_hours=(s_end - s_start) / 3600, sensed_minutes=len(v),
                   time_share_on=on.mean(), dose_share_on=p[on].sum() / p.sum() if p.sum() > 0 else np.nan,
                   mean_on=p[on].mean(), mean_off=p[~on].mean(),
                   sum_on=p[on].sum(), n_on=int(on.sum()), sum_off=p[~on].sum(), n_off=int((~on).sum()),
                   mean_moving=np.nanmean(p[sp > 5]) if (sp > 5).sum() >= 10 else np.nan,       # spec §8
                   mean_stationary=np.nanmean(p[sp < 2]) if (sp < 2).sum() >= 10 else np.nan)
        vr = v[~(v.rhflag == True)]
        onr = (vr.minute_unix >= s_start) & (vr.minute_unix <= s_end)
        rec.update(sum_on_rh=vr.pmc[onr].sum(), n_on_rh=int(onr.sum()),
                   sum_off_rh=vr.pmc[~onr].sum(), n_off_rh=int((~onr).sum()))
        pdays.append(rec)
E2d = pd.DataFrame(pdays)
E2d["month"] = pd.to_datetime(E2d.workday).dt.month
E2d["period"] = np.where(E2d.month.isin([12, 1]), "Dec-Jan", "Feb-Apr")
E2d["on_off_ratio"] = E2d.mean_on / E2d.mean_off
E2d["excess_share"] = E2d.dose_share_on - E2d.time_share_on
E2d["move_stat_ratio"] = E2d.mean_moving / E2d.mean_stationary
ABS2 = {c: f"{c}_{LABEL_E2}" for c in ["mean_on", "mean_off", "sum_on", "sum_off", "mean_moving", "mean_stationary"]}
E2d.drop(columns=["sum_on_rh", "n_on_rh", "sum_off_rh", "n_off_rh"]).rename(columns=ABS2).round(4) \
    .to_csv(os.path.join(OUT, "tables", "e2_person_days.csv"), index=False)


def e2_stats(d, key, offset=0.0, rh=False, do_boot=True):
    if len(d) == 0:
        return dict(n_days=0)
    so, no, sf, nf = (("sum_on_rh", "n_on_rh", "sum_off_rh", "n_off_rh") if rh
                      else ("sum_on", "n_on", "sum_off", "n_off"))
    d = d[(d[no] >= 30) & (d[nf] >= 30)]
    if len(d) == 0:
        return dict(n_days=0)
    on = (d[so] + offset * d[no]) / d[no]
    off = (d[sf] + offset * d[nf]) / d[nf]
    out = dict(n_days=len(d), n_persons=d.person.nunique(),
               pooled_ratio=((d[so] + offset * d[no]).sum() / d[no].sum()) / ((d[sf] + offset * d[nf]).sum() / d[nf].sum()),
               median_ratio=float(np.median(on / off)), q1_ratio=float(np.percentile(on / off, 25)),
               q3_ratio=float(np.percentile(on / off, 75)), share_on_higher=float(np.mean(on > off)),
               median_time_share=float(d.time_share_on.median()), median_dose_share=float(d.dose_share_on.median()),
               median_excess=float(d.excess_share.median()), median_shift_h=float(d.shift_hours.median()),
               median_move_stat=float(np.nanmedian(d.move_stat_ratio)) if d.move_stat_ratio.notna().any() else np.nan,
               n_move_stat=int(d.move_stat_ratio.notna().sum()))
    if do_boot and len(d) >= 10:                        # SF4: offices too when >= 10 person-days
        A = d.assign(so_=d[so] + offset * d[no], sf_=d[sf] + offset * d[nf], hi=(on > off).astype(float), one=1.0)
        D = A.groupby("workday")[["so_", no, "sf_", nf, "hi", "one"]].sum().values
        S = D[boot_idx(len(D), "E2|" + key)].sum(axis=1)
        out["pooled_ratio_ci"] = ci((S[:, 0] / S[:, 1]) / (S[:, 2] / S[:, 3]))
        out["share_on_higher_ci"] = ci(S[:, 4] / S[:, 5])
    return out


e2_rows = []
for c in ["rider", "office"]:
    for city in ["BKK", "CMI", "both"]:
        d = E2d[E2d.cls == c]
        if city != "both":
            d = d[d.city == city]
        for per in ["all", "Dec-Jan", "Feb-Apr"]:
            dd = d if per == "all" else d[d.period == per]
            for an, dsub in ([("primary", dd), ("20-slot", dd[dd.slot20])] if c == "rider" else [("primary", dd)]):
                s = e2_stats(dsub, f"{c}|{city}|{per}")
                s.update(cls=c, city=city, period=per, analysis=an)
                e2_rows.append(s)
        if c == "rider":
            for an, kw in [(RH_NOTE, dict(rh=True)), ("S-int -5", dict(offset=-5.0)), ("S-int +5", dict(offset=5.0))]:
                s = e2_stats(d, f"{c}|{city}|all", **kw)
                s.update(cls=c, city=city, period="all", analysis=an)
                e2_rows.append(s)
E2 = pd.DataFrame(e2_rows)
E2c = E2.copy()
for k in ["pooled_ratio_ci", "share_on_higher_ci"]:
    E2c[k + "_lo"] = E2c[k].map(lambda t: t[0] if isinstance(t, tuple) else np.nan)
    E2c[k + "_hi"] = E2c[k].map(lambda t: t[1] if isinstance(t, tuple) else np.nan)
    E2c = E2c.drop(columns=k)
E2c.round(4).to_csv(os.path.join(OUT, "tables", "e2_summary.csv"), index=False)
E2p = E2d.groupby(["person", "cls", "city"]).agg(n_days=("workday", "size"),
                                                  median_on_off=("on_off_ratio", "median"),
                                                  share_on_higher=("on_off_ratio", lambda s: float((s > 1).mean())),
                                                  median_excess_share=("excess_share", "median")).reset_index()
E2p["slot20"] = ~E2p.person.isin(SLOT20_EXCLUDE)
E2p.round(3).to_csv(os.path.join(OUT, "tables", "e2_by_person.csv"), index=False)


# ======================================================================== E3
def e3_daily(Rsub, min_h=4):
    rd = Rsub.groupby(["city", "date", "node"]).agg(v=("pm", "mean"), h=("pm", "size")).reset_index()
    rd = rd[rd.h >= min_h]
    cd = rd.groupby(["date", "city"]).agg(v=("v", "median"), n=("v", "size")).reset_index()
    w = cd.pivot(index="date", columns="city", values=["v", "n"])
    w.columns = [f"{a}_{b}" for a, b in w.columns]
    w = w.dropna()
    w = w[(w.n_BKK >= 2) & (w.n_CMI >= 2)].reset_index()
    w["month"] = pd.to_datetime(w.date).dt.month
    return w


def e3_stats(w, key, offset=0.0, do_boot=True):
    if len(w) == 0:
        return dict(n_dates=0)
    c, b = w.v_CMI.values + offset, w.v_BKK.values + offset
    out = dict(n_dates=len(w), ratio=c.mean() / b.mean(), median_daily_ratio=float(np.median(c / b)),
               share_cmi_higher=float(np.mean(c > b)), cmi_mean=c.mean(), bkk_mean=b.mean())
    if do_boot and len(w) >= MIN_CLUSTERS_CI:
        idx = boot_idx(len(w), "E3|" + key)
        out["ratio_ci"] = ci(c[idx].mean(1) / b[idx].mean(1))
        out["share_ci"] = ci((c[idx] > b[idx]).mean(1))
    return out


W = e3_daily(R)
W20 = e3_daily(R20)
Wout = W.merge(W20[["date", "v_BKK", "v_CMI", "n_BKK", "n_CMI"]].rename(
    columns={"v_BKK": "v20_BKK", "v_CMI": "v20_CMI", "n_BKK": "n20_BKK", "n_CMI": "n20_CMI"}), on="date", how="left")
Wout.round(3).rename(columns={"v_BKK": f"bkk_rider_median_{LABEL}", "v_CMI": f"cmi_rider_median_{LABEL}",
                              "n_BKK": "bkk_riders", "n_CMI": "cmi_riders",
                              "v20_BKK": f"bkk_rider_median_20slot_{LABEL}", "v20_CMI": f"cmi_rider_median_20slot_{LABEL}",
                              "n20_BKK": "bkk_riders_20slot", "n20_CMI": "cmi_riders_20slot"}) \
    .to_csv(os.path.join(OUT, "tables", "e3_city_daily.csv"), index=False)
e3_rows = []
for per, ms in [("all", [12, 1, 2, 3, 4]), ("Dec-Jan", [12, 1]), ("Feb-Apr", [2, 3, 4])] + \
               [(MONTHS[m], [m]) for m in [12, 1, 2, 3, 4]]:
    for an, ww in [("primary", W), ("20-slot", W20)]:
        s = e3_stats(ww[ww.month.isin(ms)], per); s.update(period=per, analysis=an); e3_rows.append(s)
for an, ww, off in [(RH_NOTE, e3_daily(R[~R.rhflag]), 0.0), ("S-int -5", W, -5.0), ("S-int +5", W, 5.0),
                    ("S-8h", e3_daily(R, 8), 0.0)]:
    s = e3_stats(ww, "all", off); s.update(period="all", analysis=an); e3_rows.append(s)
E3 = pd.DataFrame(e3_rows)


def pct_change(ww):
    base = ww[ww.month.isin([12, 1])]
    out = {}
    for mo in [12, 1, 2, 3, 4]:
        wm = ww[ww.month == mo]
        if len(wm) and len(base):
            out[mo] = (len(wm), 100 * (wm.v_CMI.mean() / base.v_CMI.mean() - 1), 100 * (wm.v_BKK.mean() / base.v_BKK.mean() - 1))
    return out


pc, pc20 = pct_change(W), pct_change(W20)
PCH = pd.DataFrame([dict(month=MONTHS[mo], n_dates=pc[mo][0], cmi_pct_change_vs_decjan=pc[mo][1],
                         bkk_pct_change_vs_decjan=pc[mo][2],
                         n_dates_20slot=pc20.get(mo, (np.nan,) * 3)[0],
                         cmi_pct_change_vs_decjan_20slot=pc20.get(mo, (np.nan,) * 3)[1],
                         bkk_pct_change_vs_decjan_20slot=pc20.get(mo, (np.nan,) * 3)[2]) for mo in pc])
E3c = E3.copy().rename(columns={"cmi_mean": f"cmi_mean_{LABEL}", "bkk_mean": f"bkk_mean_{LABEL}"})
for k in ["ratio_ci", "share_ci"]:
    E3c[k + "_lo"] = E3c[k].map(lambda t: t[0] if isinstance(t, tuple) else np.nan)
    E3c[k + "_hi"] = E3c[k].map(lambda t: t[1] if isinstance(t, tuple) else np.nan)
    E3c = E3c.drop(columns=k)
E3c.round(4).to_csv(os.path.join(OUT, "tables", "e3_summary.csv"), index=False)
PCH.round(1).to_csv(os.path.join(OUT, "tables", "e3_monthly_pct_change.csv"), index=False)
Re = R[R.date.isin(set(W.date))]
di = []
for city in ["BKK", "CMI"]:
    for per in ["all", "Dec-Jan", "Feb-Apr"]:
        x = Re[Re.city == city]
        if per != "all":
            x = x[x.period == per]
        if len(x) == 0:
            continue
        mu = x.pm.mean()
        g = x.groupby("clock").agg(m=("pm", "mean"), n=("pm", "size")).reset_index()
        for _, r in g.iterrows():
            di.append(dict(city=city, period=per, clock=int(r.clock), pct_of_mean=100 * r.m / mu, n_node_hours=int(r.n)))
DI = pd.DataFrame(di)
DI.round(1).to_csv(os.path.join(OUT, "tables", "e3_diurnal.csv"), index=False)

# ===================================================================== figures
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#d9d8d4", "#fcfcfb"
COL = {"BKK": "#2a78d6", "CMI": "#eb6834"}


def svg(w, h, body, title):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" '
            f'font-family="system-ui, sans-serif" font-size="12"><rect width="{w}" height="{h}" fill="{SURF}"/>'
            f'<text x="16" y="22" font-size="14" font-weight="600" fill="{INK}">{title}</text>{body}</svg>')


def fig_e1():
    rows = E1[(E1.analysis == "primary") & (E1.period.isin(["all", "Dec-Jan", "Feb-Apr"]))].copy()
    rows = rows[rows.n_hours > 0]
    labels = []
    for _, r in rows.iterrows():
        g = "pooled*" if (r.group == "pooled" and r.city == "CMI") else r.group
        labels.append((f"{g} {r.city} · {r.period}", r.ratio, r.get("ratio_ci"), r.city))
    W_, rowh, x0, x1 = 720, 22, 230, 690
    H_ = 60 + rowh * len(labels) + 56
    lo, hi = 0.5, 3.0
    X = lambda v: x0 + (np.log(v) - np.log(lo)) / (np.log(hi) - np.log(lo)) * (x1 - x0)
    b = ""
    for t in [0.5, 0.75, 1, 1.5, 2, 3]:
        b += f'<line x1="{X(t):.1f}" x2="{X(t):.1f}" y1="44" y2="{H_-52}" stroke="{GRID if t != 1 else INK2}" stroke-width="1"/>'
        b += f'<text x="{X(t):.1f}" y="{H_-36}" text-anchor="middle" fill="{INK2}">{t}</text>'
    b += f'<text x="{(x0+x1)/2}" y="{H_-20}" text-anchor="middle" fill="{INK2}">rider / office ratio of means (log scale; 1 = equal)</text>'
    b += f'<text x="16" y="{H_-4}" fill="{INK2}" font-size="11">* CMI pooled = O04 + O01 Chiang Mai trip hours. No CI with fewer than 5 dates.</text>'
    for i, (lab, v, c, city) in enumerate(labels):
        y = 56 + i * rowh
        b += f'<text x="{x0-8}" y="{y+4}" text-anchor="end" fill="{INK}">{lab}</text>'
        if isinstance(c, tuple) and np.isfinite(c[0]):
            b += f'<line x1="{X(max(c[0],lo)):.1f}" x2="{X(min(c[1],hi)):.1f}" y1="{y}" y2="{y}" stroke="{COL[city]}" stroke-width="2"/>'
        if np.isfinite(v):
            b += (f'<circle cx="{X(min(max(v,lo),hi)):.1f}" cy="{y}" r="5" fill="{COL[city]}" stroke="{SURF}" stroke-width="2">'
                  f'<title>{lab}: {v:.2f}</title></circle>')
    open(os.path.join(OUT, "figures", "e1_rider_office_ratio.svg"), "w").write(
        svg(W_, H_, b, f"E1 Rider vs office personal PM2.5 ({LABEL} sensor), matched city-hours, 95% CI over dates"))


def fig_e2():
    d = E2d[E2d.cls == "rider"]
    W_, H_, x0, x1, y0, y1 = 720, 300, 60, 690, 260, 50
    bins = np.linspace(np.log(0.25), np.log(4), 25)
    b = ""
    maxc = 1
    hs = {}
    for city in ["BKK", "CMI"]:
        v = np.log(d[d.city == city].on_off_ratio.dropna().clip(0.26, 3.9))
        hs[city] = np.histogram(v, bins)[0]
        maxc = max(maxc, hs[city].max())
    X = lambda lv: x0 + (lv - bins[0]) / (bins[-1] - bins[0]) * (x1 - x0)
    for t in [0.25, 0.5, 1, 2, 4]:
        b += f'<line x1="{X(np.log(t)):.1f}" x2="{X(np.log(t)):.1f}" y1="{y1}" y2="{y0}" stroke="{GRID if t != 1 else INK2}"/>'
        b += f'<text x="{X(np.log(t)):.1f}" y="{y0+16}" text-anchor="middle" fill="{INK2}">{t}</text>'
    for city in ["BKK", "CMI"]:
        pts = []
        for i, c in enumerate(hs[city]):
            xm = X((bins[i] + bins[i + 1]) / 2)
            pts.append(f"{xm:.1f},{y0 - c / maxc * (y0 - y1):.1f}")
        b += f'<polyline points="{" ".join(pts)}" fill="none" stroke="{COL[city]}" stroke-width="2"/>'
    b += (f'<text x="{x1}" y="{y1}" text-anchor="end" fill="{INK}"><tspan fill="{COL["BKK"]}">■</tspan> BKK  '
          f'<tspan fill="{COL["CMI"]}">■</tspan> CMI</text>')
    b += f'<text x="{(x0+x1)/2}" y="{H_-8}" text-anchor="middle" fill="{INK2}">on-shift / off-shift mean PM, same sensor, per rider-day (log scale)</text>'
    open(os.path.join(OUT, "figures", "e2_on_off_shift.svg"), "w").write(
        svg(W_, H_, b, f"E2 On-shift vs off-shift exposure per rider-day ({LABEL_E2}); count of rider-days"))


def fig_e3():
    W_, H_ = 720, 560
    b = ""
    x0, x1, y0, y1 = 60, 690, 250, 50
    dts = pd.to_datetime(W.date)
    t0, t1 = pd.Timestamp("2025-12-10"), pd.Timestamp("2026-05-01")
    X = lambda t: x0 + (t - t0) / (t1 - t0) * (x1 - x0)
    lo, hi = 0.25, 8
    Y = lambda v: y0 - (np.log(v) - np.log(lo)) / (np.log(hi) - np.log(lo)) * (y0 - y1)
    for t in [0.25, 0.5, 1, 2, 4, 8]:
        b += f'<line x1="{x0}" x2="{x1}" y1="{Y(t):.1f}" y2="{Y(t):.1f}" stroke="{GRID if t != 1 else INK2}"/>'
        b += f'<text x="{x0-6}" y="{Y(t)+4:.1f}" text-anchor="end" fill="{INK2}">{t}</text>'
    for mo in ["2026-01-01", "2026-02-01", "2026-03-01", "2026-04-01"]:
        t = pd.Timestamp(mo)
        b += f'<text x="{X(t):.1f}" y="{y0+16}" text-anchor="middle" fill="{INK2}">{t.strftime("%b")}</text>'
    for t, r, d in zip(dts, (W.v_CMI / W.v_BKK).clip(lo, hi), W.date):
        b += f'<circle cx="{X(t):.1f}" cy="{Y(r):.1f}" r="4" fill="{COL["CMI"]}" stroke="{SURF}" stroke-width="2"><title>{d}: {r:.2f}</title></circle>'
    b += f'<text x="{x0}" y="42" fill="{INK2}">A. Daily CMI / BKK ratio of rider medians, same dates (log scale; 1 = equal)</text>'
    x0, x1, y0, y1 = 60, 690, 520, 310
    X = lambda h: x0 + h / 23 * (x1 - x0)
    lo, hi = 40, 200
    Y = lambda v: y0 - (v - lo) / (hi - lo) * (y0 - y1)
    for t in [50, 100, 150, 200]:
        b += f'<line x1="{x0}" x2="{x1}" y1="{Y(t):.1f}" y2="{Y(t):.1f}" stroke="{GRID if t != 100 else INK2}"/>'
        b += f'<text x="{x0-6}" y="{Y(t)+4:.1f}" text-anchor="end" fill="{INK2}">{t}%</text>'
    for h in [0, 6, 12, 18, 23]:
        b += f'<text x="{X(h):.1f}" y="{y0+16}" text-anchor="middle" fill="{INK2}">{h:02d}:00</text>'
    for city in ["BKK", "CMI"]:
        x = DI[(DI.city == city) & (DI.period == "all")].sort_values("clock")
        pts = " ".join(f"{X(h):.1f},{Y(min(max(v, lo), hi)):.1f}" for h, v in zip(x.clock, x.pct_of_mean))
        b += f'<polyline points="{pts}" fill="none" stroke="{COL[city]}" stroke-width="2"/>'
        if len(x):
            b += f'<text x="{x1+4}" y="{Y(min(max(x.pct_of_mean.iloc[-1], lo), hi)) + (-4 if city == "CMI" else 10):.1f}" fill="{INK}" font-size="11">{city}</text>'
    b += f'<text x="{x0}" y="302" fill="{INK2}">B. Hour-of-day rider PM as % of the city mean, same dates</text>'
    open(os.path.join(OUT, "figures", "e3_city_sameday.svg"), "w").write(
        svg(W_ + 30, H_, b, f"E3 Chiang Mai vs Bangkok, same days ({LABEL} sensor)"))


fig_e1(); fig_e2(); fig_e3()

# ===================================================================== reports
RUNARGS = ("--cal-json " + args.cal_json) if CAL is not None else ("--pm-col " + args.pm_col)
RUNARGS += " --include-6006" if args.include_6006 else ""


def ci_s(c, f=fmt):
    return f" [{f(c[0])}, {f(c[1])}]" if isinstance(c, tuple) and np.isfinite(c[0]) else ""


def e1_line(r):
    return (f"| {r.group} | {r.city} | {r.period} | {r.analysis} | {int(r.n_hours)} | {int(r.n_dates)} | "
            f"{fmt(r.med_riders, 0)} | {fmt(r.ratio)}{ci_s(r.get('ratio_ci'))} | {fmt(r.med_hourly_ratio)} | "
            f"{pct(r.share_higher)}{ci_s(r.get('share_ci'), pct)} |")


HDR = (f"*Generated by `scripts/calib_run_all.py` ({RUNARGS}), v3 2026-11-11, under `calibration/RELATIVE_SPEC.md` "
       f"(approved 2026-11-10; amendments §7 node 6006 {'kept' if args.include_6006 else 'ignored'}, §8 implementation rules and "
       f"audit fixes, §9 20-slot sensitivity). Uncalibrated low-cost sensor readings; relative contrasts only; calibration by AIT "
       f"in progress. Absolute values are in `CAL_PENDING.md` only. **Provisional until S7 re-audits.** Not paper prose.*")
L = [f"# EXPOSURE_GAP.md: E1 rider vs office, E2 on- vs off-shift (E1 {LABEL}, E2 {LABEL_E2} sensor PM2.5)", "", HDR, "",
     "## E1 Rider vs office, matched city-hours (≥ 2 rider nodes that hour)", "",
     "Ratio = Σ rider comparator / Σ office over matched hours; 95% CI by bootstrapping calendar dates (no CI with < 5 dates). "
     "Share = % of matched hours rider comparator > office. **20-slot** = the same analysis without the two replacement riders "
     "R12 and R14 (spec §9; sensitivity, decided after the primary results). Bangkok has no replacements, so its 20-slot rows equal the primary.", "",
     "**Chiang Mai `pooled` = O04 (311 h) plus O01's 22 h on 4 dates when O01's sensor was in Chiang Mai (GPS-city rule, spec §2; "
     "2 of the 4 dates look like travel hours).** O04 alone is the `O04` row and the `S-O04-only` row. Which is primary is an open user "
     "decision (audit SF1).", "",
     "| Office | City | Period | Analysis | Hours | Dates | Rider nodes/h (median) | Ratio of means [CI] | Median hourly ratio | % hours rider higher [CI] |",
     "|---|---|---|---|---|---|---|---|---|---|"]
for _, r in E1[E1.analysis.isin(["primary", "20-slot"]) & (E1.n_hours > 0)].iterrows():
    L.append(e1_line(r))
    if r.analysis == "primary" and r.group == "pooled" and r.period == "all":
        calp(f"E1 {r.city} pooled", "office mean", r.office_mean)
        calp(f"E1 {r.city} pooled", "rider comparator mean", r.rider_mean)
        calp(f"E1 {r.city} pooled", "rider − office mean difference", r["diff"])
    elif r.analysis == "primary" and r.period == "all":
        calp(f"E1 {r.group} {r.city}", "rider − office mean difference", r["diff"])
L += ["", "### Sensitivity checks (pooled, all months)", "",
      "Pre-set in the spec: S-RH, S-int, S-daytime, S-LORO. Added later (labelled): S-O04-only (audit SF1), S-incl-6006 (§7), "
      "S-zero-min (exploratory, audit N1: hourly means rebuilt without exact-zero minutes, which are common on office sensors).", "",
      "| City | Check | Hours | Dates | Ratio of means [CI] | % hours rider higher [CI] | LORO range |", "|---|---|---|---|---|---|---|"]
for _, r in E1[~E1.analysis.isin(["primary", "20-slot"])].iterrows():
    if r.analysis == "S-LORO":
        L.append(f"| {r.city} | leave one rider node out | | | | | {fmt(r.loro_min)} – {fmt(r.loro_max)} |")
    else:
        L.append(f"| {r.city} | {r.analysis} | {int(r.n_hours)} | {int(r.n_dates)} | {fmt(r.ratio)}{ci_s(r.get('ratio_ci'))} | "
                 f"{pct(r.share_higher)}{ci_s(r.get('share_ci'), pct)} | |")
if not NZ_OK:
    L.append("| both | S-zero-min | not run: needs --cal-json (minute data) | | | | |")


def e2_line(r):
    if r.n_days == 0:
        return f"| {r.cls} | {r.city} | {r.period} | {r.analysis} | 0 | | | | | | |"
    return (f"| {r.cls} | {r.city} | {r.period} | {r.analysis} | {int(r.n_days)} ({int(r.n_persons)}) | "
            f"{fmt(r.pooled_ratio)}{ci_s(r.get('pooled_ratio_ci'))} | {fmt(r.median_ratio)} ({fmt(r.q1_ratio)}–{fmt(r.q3_ratio)}) | "
            f"{pct(r.share_on_higher)}{ci_s(r.get('share_on_higher_ci'), pct)} | "
            f"{pct(r.median_time_share)} / {pct(r.median_dose_share)} | {fmt(r.median_shift_h,1)} | {fmt(r.median_move_stat)} ({r.n_move_stat}) |")


L += ["", f"## E2 The job sets the dose: on-shift vs off-shift, same sensor, per person-workday ({LABEL_E2})", "",
      "Shift = first to last sustained-movement minute (S1 QC_RULES §D, mirrored for the end). Person-day rules (spec §3 and §8): "
      "≥ 12 valid PM hours; ≥ 30 minutes **with a GPS speed value**; shift ≥ 1 h; ≥ 30 sensed minutes both on and off shift. "
      "Moving/stationary ratio needs ≥ 10 minutes in each class. City = GPS city of the day (most valid hours). "
      "CI: bootstrap over workdays (clusters), riders and offices with ≥ 10 person-days.", "",
      "| Class | City | Period | Analysis | Person-days (persons) | Pooled on/off ratio [CI] | Median day ratio (IQR) | % days on > off [CI] | Median time share / dose share on shift | Median shift h | Median moving/stationary ratio (days) |",
      "|---|---|---|---|---|---|---|---|---|---|---|"]
for _, r in E2.iterrows():
    L.append(e2_line(r))
L += ["", "Office moving/stationary ratios are distorted by exact-zero minutes on office sensors (audit N1); treat them as unreliable.", "",
      "Per-person medians: `tables/e2_by_person.csv`. Person-day rows: `tables/e2_person_days.csv` (column `slot20` marks the 20-slot set).", "",
      "Figures: `figures/e1_rider_office_ratio.svg`, `figures/e2_on_off_shift.svg`."]
open(os.path.join(OUT, "EXPOSURE_GAP.md"), "w").write("\n".join(L) + "\n")


def e3_line(r):
    if r.n_dates == 0:
        return f"| {r.period} | {r.analysis} | 0 | | | |"
    return (f"| {r.period} | {r.analysis} | {int(r.n_dates)} | {fmt(r.ratio)}{ci_s(r.get('ratio_ci'))} | "
            f"{fmt(r.median_daily_ratio)} | {pct(r.share_cmi_higher)}{ci_s(r.get('share_ci'), pct)} |")


L = [f"# CITY_SAMEDAY.md: E3 Chiang Mai vs Bangkok on the same dates ({LABEL} sensor PM2.5)", "", HDR, "",
     "COMPASS '25 already reported city and diurnal patterns: context, not a lead.", "",
     "Eligible date = ≥ 2 riders with ≥ 4 valid hours in **both** cities. City daily value = median of rider daily means "
     "(each rider-day = mean of its valid hourly means). CI = bootstrap over dates (no CI with < 5 dates). "
     "**20-slot** = without replacement riders R12 and R14 (spec §9).", "",
     "| Period | Analysis | Dates | CMI/BKK ratio of means [CI] | Median daily ratio | % dates CMI > BKK [CI] |", "|---|---|---|---|---|---|"]
for _, r in E3.iterrows():
    L.append(e3_line(r))
    if r.analysis == "primary" and r.period == "all":
        calp("E3 all dates", "CMI mean of city daily values", r.cmi_mean)
        calp("E3 all dates", "BKK mean of city daily values", r.bkk_mean)
L += ["", "## % change of each city's mean daily value vs its own Dec–Jan baseline (same eligible dates)", "",
      "| Month | Dates | CMI % change | BKK % change | Dates (20-slot) | CMI (20-slot) | BKK (20-slot) |", "|---|---|---|---|---|---|---|"]
for _, r in PCH.iterrows():
    L.append(f"| {r.month} | {int(r.n_dates)} | {spct(r.cmi_pct_change_vs_decjan)} | {spct(r.bkk_pct_change_vs_decjan)} | "
             f"{'' if not np.isfinite(r.n_dates_20slot) else int(r.n_dates_20slot)} | {spct(r.cmi_pct_change_vs_decjan_20slot)} | "
             f"{spct(r.bkk_pct_change_vs_decjan_20slot)} |")
L += ["", "## Diurnal profile (rider node-hours on eligible dates, % of the city mean; all 22 riders)", "",
      "| Clock hour | " + " | ".join(f"{c} {p}" for c in ["BKK", "CMI"] for p in ["all", "Dec-Jan", "Feb-Apr"]) + " |",
      "|---" * 7 + "|"]
piv = DI.pivot_table(index="clock", columns=["city", "period"], values="pct_of_mean")
for h in range(24):
    vals = []
    for c in ["BKK", "CMI"]:
        for p in ["all", "Dec-Jan", "Feb-Apr"]:
            v = piv.loc[h, (c, p)] if (h in piv.index and (c, p) in piv.columns) else np.nan
            vals.append("n/a" if not np.isfinite(v) else f"{v:.0f}%")
    L.append(f"| {h:02d} | " + " | ".join(vals) + " |")
L += ["", "Tables: `tables/e3_city_daily.csv` (includes 20-slot columns), `tables/e3_summary.csv`, `tables/e3_monthly_pct_change.csv`, "
      "`tables/e3_diurnal.csv`. Figure: `figures/e3_city_sameday.svg`."]
open(os.path.join(OUT, "CITY_SAMEDAY.md"), "w").write("\n".join(L) + "\n")

# CAL_PENDING
L = ["# CAL_PENDING.md: absolute values awaiting AIT calibration (S2)", "",
     f"*Generated by `scripts/calib_run_all.py` ({LABEL}). Every absolute PM value S2 quotes appears here as `[CAL: raw value]`, in raw sensor units. "
     "Rerun with `--cal-json calibration/ait_equation.json` on Nov 28 to replace them. Never compare these with WHO 15 / Thai 37.5 µg/m³ before then.*", "",
     f"{len(CALP)} values.", "",
     "| Where | Quantity | Value |", "|---|---|---|"]
for w, q, v in CALP:
    L.append(f"| {w} | {q} | [CAL: {v:.1f}] |")
L += ["", "Absolute-value columns in the tables (all suffixed `_raw` or `_calibrated`): `e1_matched_hours.csv` (office_*, rider_mean_*), "
      "`e1_summary.csv` (office_mean_*, rider_mean_*, rider_minus_office_*), `e2_person_days.csv` (mean_on_*, mean_off_*, sum_on_*, "
      "sum_off_*, mean_moving_*, mean_stationary_*), `e3_city_daily.csv` (*_rider_median_*), `e3_summary.csv` (cmi_mean_*, bkk_mean_*).", "",
     "Deferred to Gate 4 (not computed on raw data, by rule R5): share of riders' sensing hours above WHO 15 / Thai 37.5 µg/m³, by city; personal vs station (E4)."]
open(os.path.join(OUT, "CAL_PENDING.md"), "w").write("\n".join(L) + "\n")
print("done", LABEL, "E1 rows", len(E1), "E2 person-days", len(E2d), "E3 dates", len(W), "20-slot E3 dates", len(W20))
