"""S1 Tasks 4-5: QC flags (rules fixed in foundation/QC_RULES.md, approved 2026-11-09) and
workday (fatigue) features. NO PM2.5 / temperature / humidity used anywhere in this file.

Outputs (foundation/): qc_session_flags.csv, qc_pef_flags.csv, qc_rider_summary.csv,
outlier_register.csv, workday_features_pvt.csv, workday_features_pef.csv,
feature_correlations.csv, qc_timeline.svg
"""
import os, sys, ast
import numpy as np, pandas as pd
import warnings; warnings.filterwarnings("ignore", category=FutureWarning)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from foundation_common import *
from foundation_ids import person_table, assignments
from foundation_cohort_const import OFFICE_SENSOR

W = os.path.join(FOUND, "_work")
pt = person_table(); asg = assignments()
pid_of = dict(zip(pt.user, pt.pid)); role_of = dict(zip(pt.pid, pt.role))
win = {r.pid: (r.start, r.end) for r in pt.itertuples() if r.role == "rider"}


def node_for(pid, date):
    if role_of[pid] == "office":
        return OFFICE_SENSOR[[u for u, p in pid_of.items() if p == pid][0]][0]
    a = asg[(asg.pid == pid) & (asg.start <= date) & (asg.end >= date)]
    return int(a.sensor.iloc[0]) if len(a) else np.nan


def load_tests():
    p = pd.read_csv(os.path.join(DATA, "pvt.csv")); f = pd.read_csv(os.path.join(DATA, "peakflow.csv"))
    for d in (p, f):
        d["user"] = d.username.str.strip(); d.drop(d[~d.user.isin(pid_of)].index, inplace=True)
        d["pid"] = d.user.map(pid_of); d["role"] = d.pid.map(role_of)
        d["local"] = pd.to_datetime(d.timestamp); d["date"] = d.local.dt.normalize()
    p["start_local"] = p.local - pd.Timedelta(seconds=300)   # timestamp = END of 5-min test
    p["t_unix"] = local_to_unix(p.start_local); p["end_unix"] = local_to_unix(p.local)
    f["t_unix"] = local_to_unix(f.local)
    for d, tcol in ((p, "start_local"), (f, "local")):
        d["workday"] = (d[tcol] - pd.Timedelta(hours=4)).dt.normalize()
        d["min_after4"] = ((d[tcol] - d[tcol].dt.normalize()).dt.total_seconds() / 60 - 240) % 1440
        d["node"] = [node_for(a, b) for a, b in zip(d.pid, d.date)]
    return p.sort_values(["pid", "local"]).reset_index(drop=True), f.sort_values(["pid", "local"]).reset_index(drop=True)


def rsd(x):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    return 1.4826 * np.median(np.abs(x - np.median(x))) if len(x) else np.nan


# ---------------------------------------------------------------- minutes / workdays
MIN = {}
def minutes(node):
    if node not in MIN:
        d = pd.read_csv(os.path.join(W, f"minute_{node}.csv"), usecols=["minute_unix", "n_rows", "n_gps", "speed_kmh", "km_step"])
        d["moving"] = d.speed_kmh > 5
        d["wd"] = (unix_to_local(d.minute_unix) - pd.Timedelta(hours=4)).normalize()
        MIN[node] = d
    return MIN[node]

WD = {}
def workday_info(node, wd):
    key = (node, wd)
    if key in WD: return WD[key]
    d = minutes(node); g = d[d.wd == wd]
    info = dict(first_data=np.nan, shift_start=np.nan, last_activity=np.nan, gps_minutes=0)
    if len(g):
        info["first_data"] = g.minute_unix.min()
        info["gps_minutes"] = int((g.n_gps > 0).sum())
        if info["gps_minutes"] >= 30:
            u0 = local_to_unix(wd) + 4 * 3600
            grid = np.zeros(1440 + 15, bool)
            idx = ((g.minute_unix - u0) // 60).astype(int).values
            ok = (idx >= 0) & (idx < 1440)
            grid[idx[ok]] = g.moving.values[ok]
            cs = np.concatenate([[0], np.cumsum(grid)])
            fwd = cs[15:1440 + 15] - cs[:1440]          # moving count in [m, m+15)
            starts = np.where((fwd >= 8) & grid[:1440])[0]
            if len(starts):
                info["shift_start"] = u0 + 60 * starts[0]
                back = np.array([cs[m + 1] - cs[max(0, m - 14)] for m in range(1440)])  # moving in [m-14, m]
                ends = np.where((back >= 8) & grid[:1440])[0]
                info["last_activity"] = u0 + 60 * ends[-1] if len(ends) else np.nan
    WD[key] = info
    return info


def features(df, tcol, kind):
    out = []
    for r in df.itertuples():
        row = dict(idx=r.Index)
        if np.isnan(r.node):
            out.append(row); continue
        node = int(r.node); t = getattr(r, tcol)
        wi = workday_info(node, r.workday)
        prev = workday_info(node, r.workday - pd.Timedelta(days=1))
        d = minutes(node)
        sel = (d.wd == r.workday) & (d.minute_unix < t)
        row["km_ridden_before_test"] = float(d.loc[sel & d.moving, "km_step"].sum()) if wi["gps_minutes"] >= 30 else np.nan
        row["sensor_on_hours_before_test"] = (t - wi["first_data"]) / 3600 if not np.isnan(wi["first_data"]) and wi["first_data"] <= t else np.nan
        ss = wi["shift_start"]
        row["shift_start_local"] = str(unix_to_local([ss])[0])[:16] if not np.isnan(ss) else ""
        row["test_before_shift"] = bool(not np.isnan(ss) and ss > t)
        row["hours_on_shift_before_test"] = max(0.0, (t - ss) / 3600) if not np.isnan(ss) else np.nan
        row["rest_gap_hours"] = (ss - prev["last_activity"]) / 3600 if not (np.isnan(ss) or np.isnan(prev["last_activity"])) else np.nan
        row["gps_minutes_workday"] = wi["gps_minutes"]
        out.append(row)
    F = pd.DataFrame(out).set_index("idx")
    base = df.join(F)
    tl = base["start_local"] if kind == "pvt" else base["local"]
    base["clock_hour"] = tl.dt.hour + tl.dt.minute / 60
    base["day_of_week"] = tl.dt.dayofweek
    base["tests_already_today"] = base.groupby(["pid", "workday"]).cumcount()
    st = {p: w[0] for p, w in win.items()}
    base["days_since_enrolment"] = (base.date - base.pid.map(st)).dt.days
    return base


# ---------------------------------------------------------------- PVT flags
def pvt_flags(p):
    rows = []
    for r in p.itertuples():
        rt = np.asarray(ast.literal_eval(r.response_times), float)
        at = np.asarray(ast.literal_eval(r.all_tap_times), float)
        rv = rt[rt >= 100]
        rows.append(dict(dur_s=at.max() / 1000 if len(at) else 0, false_starts=r.total_taps - r.valid_taps,
                         n_anticip=int((rt < 100).sum()), med_rt=np.median(rv) if len(rv) else np.nan,
                         iqr_rt=np.subtract(*np.percentile(rv, [75, 25])) if len(rv) else np.nan,
                         max_rt=rt.max() if len(rt) else np.nan, n_rt_5_10s=int(((rt >= 5000) & (rt < 10000)).sum()),
                         max_tap_gap_s=np.diff(np.sort(at)).max() / 1000 if len(at) > 1 else np.nan,
                         speed=np.mean(1000 / rv) if len(rv) else np.nan))
    q = pd.concat([p.reset_index(drop=True), pd.DataFrame(rows)], axis=1)
    q["P1_incomplete"] = (q.valid_taps < 50) | (q.dur_s < 240)
    q["P2_low_accuracy"] = q.percent_success < 80
    q["P3_false_starts"] = (q.false_starts >= 5) & ~q.P2_low_accuracy
    q["P4_anticipations"] = q.n_anticip >= 3
    g = q.groupby("pid")
    q["P5_slow_session"] = q.med_rt > g.med_rt.transform("median") + 3 * g.med_rt.transform(rsd)
    q["P6_erratic"] = q.iqr_rt > g.iqr_rt.transform("median") + 3 * g.iqr_rt.transform(rsd)
    q["P7_interruption"] = (q.max_rt >= 10000) | (q.max_tap_gap_s >= 15)
    q["P7b_long_rts_5_10s"] = q.n_rt_5_10s
    # P8 moving during test [start, end)
    mv, ns = [], []
    for r in q.itertuples():
        if np.isnan(r.node): mv.append(np.nan); ns.append(0); continue
        d = minutes(int(r.node))
        lo = (r.t_unix // 60) * 60
        s = d[(d.minute_unix >= lo) & (d.minute_unix < r.end_unix)].speed_kmh.dropna()
        ns.append(len(s)); mv.append(int((s > 5).sum()) if len(s) >= 3 else np.nan)
    q["P8_speed_minutes"] = ns
    q["P8_moving"] = pd.Series(mv).map(lambda x: np.nan if np.isnan(x) else bool(x >= 2)).values
    lo5 = g.min_after4.transform(lambda x: np.percentile(x, 5)); hi95 = g.min_after4.transform(lambda x: np.percentile(x, 95))
    q["P9_unusual_hour"] = (q.min_after4 < lo5) | (q.min_after4 > hi95)
    prev_end = q.groupby("pid").end_unix.shift()
    q["P10_near_duplicate"] = (q.t_unix - prev_end) < 600
    q["P11_outside_enrolment"] = [(r.role == "rider") and not (win[r.pid][0] <= r.date <= win[r.pid][1]) for r in q.itertuples()]
    excl = ["P1_incomplete", "P2_low_accuracy", "P5_slow_session", "P7_interruption", "P8_moving", "P10_near_duplicate", "P11_outside_enrolment"]
    q["any_exclusion"] = q[excl].fillna(False).astype(bool).any(axis=1)
    q["any_flag"] = q["any_exclusion"] | q[["P3_false_starts", "P4_anticipations", "P6_erratic", "P9_unusual_hour"]].any(axis=1)
    return q, excl


def pef_flags(f, p):
    f = f.copy()
    pb = f[f.peak_flow >= 50].groupby("pid").peak_flow.quantile(0.95)
    f["personal_best"] = f.pid.map(pb)
    f["F1_low"] = f.peak_flow < 50
    f["F2_high"] = f.peak_flow > 1.3 * f.personal_best
    # sittings: readings within 5 min of the previous reading
    f["sitting"] = (f.groupby("pid").t_unix.diff().fillna(1e9) > 300).cumsum()
    smax = f.groupby("sitting").peak_flow.transform("max"); smin = f.groupby("sitting").peak_flow.transform("min")
    ssize = f.groupby("sitting").peak_flow.transform("size")
    f["sitting_size"] = ssize
    f["F3_poor_reproducibility"] = (ssize > 1) & ((smax - smin) > 0.2 * smax)
    ends = p.groupby("pid").end_unix.apply(np.sort).to_dict()
    def after_pvt(r):
        e = ends.get(r.pid)
        if e is None: return False
        i = np.searchsorted(e, r.t_unix, "right") - 1
        return bool(i >= 0 and 0 <= r.t_unix - e[i] <= 120)
    f["F4_right_after_pvt"] = [after_pvt(r) for r in f.itertuples()]
    f["F5_outside_enrolment"] = [(r.role == "rider") and not (win[r.pid][0] <= r.date <= win[r.pid][1]) for r in f.itertuples()]
    f["any_exclusion"] = f[["F1_low", "F2_high", "F5_outside_enrolment"]].any(axis=1)
    return f


def rider_summary(q, f):
    rows = []
    for pid in [x for x in pt.pid if role_of[x] == "rider"]:
        s, e = win[pid]; wk = ((e - s).days + 1) / 7
        qp = q[q.pid == pid]; fp = f[f.pid == pid]
        qin = qp[~qp.P11_outside_enrolment]; fin = fp[~fp.F5_outside_enrolment]
        days = set(qin.date) | set(fin.date)
        tdates = sorted(days); gaps = []
        for a, b in zip(tdates[:-1], tdates[1:]):
            if (b - a).days >= 7: gaps.append(f"{a.date()}→{b.date()} ({(b - a).days} d)")
        # timing shift on PVT (minutes after 04:00)
        med_all = qin.min_after4.median()
        wkm = qin.groupby(qin.local.dt.strftime("%G-W%V")).min_after4.agg(["median", "size"])
        wkm = wkm[wkm["size"] >= 3]
        shifted = wkm[(wkm["median"] - med_all).abs() >= 240]
        late = ((qin.local.dt.hour >= 21) | (qin.local.dt.hour < 4)).mean()
        sp = qin.dropna(subset=["speed"])
        slope = ols(sp.speed, np.column_stack([np.ones(len(sp)), np.log(np.arange(1, len(sp) + 1))]))[0][1] if len(sp) >= 10 else np.nan
        rows.append(dict(pid=pid, city=pt.set_index("pid").city[pid], enrol_start=str(s.date()), enrol_end=str(e.date()),
                         enrol_weeks=round(wk, 1), n_pvt_in_window=len(qin), n_pef_in_window=len(fin),
                         pvt_per_week=round(len(qin) / wk, 2), share_days_with_test=round(len(days) / ((e - s).days + 1), 3),
                         share_sessions_exclusion=round(qp.any_exclusion.mean(), 3), n_gaps_ge7d=len(gaps), gaps_ge7d="; ".join(gaps),
                         median_test_time=f"{int(((med_all + 240) % 1440) // 60):02d}:{int((med_all + 240) % 60):02d}",
                         n_weeks_time_shift_ge4h=len(shifted), weeks_time_shift=", ".join(shifted.index),
                         share_tests_21_04=round(late, 3), practice_slope_log=slope,
                         n_P1=int(qp.P1_incomplete.sum()), n_P2=int(qp.P2_low_accuracy.sum()), n_P5=int(qp.P5_slow_session.sum()),
                         n_P7=int(qp.P7_interruption.sum()), n_P8=int(qp.P8_moving.fillna(False).astype(bool).sum()),
                         n_P8_evaluable=int(qp.P8_moving.notna().sum()), n_P10=int(qp.P10_near_duplicate.sum()),
                         n_P11=int(qp.P11_outside_enrolment.sum()), n_P3=int(qp.P3_false_starts.sum()), n_P4=int(qp.P4_anticipations.sum()),
                         n_P6=int(qp.P6_erratic.sum()), n_P9=int(qp.P9_unusual_hour.sum()),
                         n_F1=int(fp.F1_low.sum()), n_F2=int(fp.F2_high.sum()), n_F3=int(fp.F3_poor_reproducibility.sum()),
                         n_F5=int(fp.F5_outside_enrolment.sum())))
    R = pd.DataFrame(rows)
    med_rate = R.pvt_per_week.median()
    R["R1_short_enrolment"] = R.enrol_weeks < 8
    R["R2_low_compliance"] = R.pvt_per_week < 0.5 * med_rate
    R["R3_low_day_coverage"] = R.share_days_with_test < 0.25
    R["R4_many_flagged"] = R.share_sessions_exclusion >= 0.20
    R["R5_long_gaps"] = R.n_gaps_ge7d > 0
    R["R6_timing_shift"] = R.n_weeks_time_shift_ge4h >= 2
    sl = R.practice_slope_log
    R["R7_practice_outlier"] = (sl - sl.median()).abs() > 3 * rsd(sl)
    # R5b (added 2026-11-09 after approval, DESCRIPTIVE): gaps not overlapping a cohort-wide break
    # (a break = gap interval shared, by overlap, with >= 5 riders' gaps)
    allg = []
    for g in R.gaps_ge7d:
        for x in (g.split("; ") if g else []):
            a, b = x.split(" ")[0].split("→"); allg.append((pd.Timestamp(a), pd.Timestamp(b)))
    def shared(a, b):
        return sum(1 for c, d in allg if c < b and a < d) >= 5
    spec = []
    for g in R.gaps_ge7d:
        keep = []
        for x in (g.split("; ") if g else []):
            a, b = x.split(" ")[0].split("→")
            if not shared(pd.Timestamp(a), pd.Timestamp(b)): keep.append(x)
        spec.append("; ".join(keep))
    R["gaps_rider_specific"] = spec
    R["R5b_rider_specific_gaps"] = R.gaps_rider_specific != ""
    R.attrs["cohort_median_pvt_per_week"] = med_rate
    return R


def register(R):
    rows = []
    rf = ["R1_short_enrolment", "R2_low_compliance", "R3_low_day_coverage", "R4_many_flagged", "R5b_rider_specific_gaps", "R6_timing_shift", "R7_practice_outlier"]
    for r in R.itertuples():
        flags = [c.split("_", 1)[0] for c in rf if getattr(r, c)]
        sess = {k: getattr(r, "n_" + k) for k in ["P1", "P2", "P5", "P7", "P8", "P10", "P11", "F1", "F2"] if getattr(r, "n_" + k) > 0}
        if not flags and r.share_sessions_exclusion < 0.10:
            continue
        desc = [f"enrolled {r.enrol_start}–{r.enrol_end} ({r.enrol_weeks} wk)",
                f"{r.pvt_per_week} PVT/wk (cohort median {R.attrs['cohort_median_pvt_per_week']:.2f})",
                f"tests on {r.share_days_with_test:.0%} of enrolled days", f"median test time {r.median_test_time}",
                f"{r.share_tests_21_04:.0%} of PVTs between 21:00 and 04:00"]
        if r.gaps_rider_specific: desc.append(f"rider-specific gap(s) ≥ 7 d (outside cohort-wide breaks): {r.gaps_rider_specific}")
        if r.n_weeks_time_shift_ge4h: desc.append(f"weeks with median test time ≥ 4 h off own median: {r.weeks_time_shift}")
        desc.append(f"{r.share_sessions_exclusion:.0%} of PVT sessions carry an EXCLUSION flag")
        if r.n_P8: desc.append(f"{r.n_P8} of {r.n_P8_evaluable} GPS-evaluable PVTs taken while the sensor moved > 5 km/h")
        if r.n_P5: desc.append(f"{r.n_P5} sessions grossly slower than own usual")
        if r.n_P9: desc.append(f"{r.n_P9} PVTs outside own usual hours")
        rows.append(dict(pid=r.pid, city=r.city, rider_flags=",".join(flags), session_flag_counts=str(sess),
                         date_range=f"{r.enrol_start} to {r.enrol_end}", pattern="; ".join(desc)))
    return pd.DataFrame(rows)


def timeline_svg(q, f, path):
    riders = [x for x in pt.pid if role_of[x] == "rider"]
    d0, d1 = pd.Timestamp("2025-12-08"), pd.Timestamp("2026-05-02")
    ndays = (d1 - d0).days
    cw, ch, left, top, colw, gaprow = 520, 120, 46, 40, 590, 26
    ncol = 2; nrow = (len(riders) + 1) // 2
    Wd = left + ncol * colw; Ht = top + nrow * (ch + gaprow) + 40
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{Wd}" height="{Ht}" font-family="sans-serif" font-size="10">',
           '<rect width="100%" height="100%" fill="white"/>',
           f'<text x="{left}" y="18" font-size="13" font-weight="bold">Per-rider test timeline (x = date Dec 8 – May 1; y = clock time 04:00 top → 04:00 bottom)</text>',
           f'<text x="{left}" y="32" font-size="10">grey = PVT, no flag · orange = PVT descriptive flag · red = PVT exclusion flag · blue triangle = peak flow (red outline = exclusion) · shaded = outside enrolment</text>']
    for i, pid in enumerate(riders):
        cx = left + (i % ncol) * colw; cy = top + (i // ncol) * (ch + gaprow) + 14
        s, e = win[pid]
        out.append(f'<text x="{cx - 40}" y="{cy + ch / 2}" font-weight="bold">{pid}</text>')
        out.append(f'<rect x="{cx}" y="{cy}" width="{cw}" height="{ch}" fill="#eee"/>')
        xs = cx + (s - d0).days / ndays * cw; xe = cx + ((e - d0).days + 1) / ndays * cw
        out.append(f'<rect x="{xs:.1f}" y="{cy}" width="{xe - xs:.1f}" height="{ch}" fill="#fff" stroke="#bbb"/>')
        for h in (8, 12, 16, 20, 24):
            y = cy + (h - 4) / 24 * ch
            out.append(f'<line x1="{cx}" x2="{cx + cw}" y1="{y:.1f}" y2="{y:.1f}" stroke="#ddd"/>')
            if i % ncol == 0: out.append(f'<text x="{cx - 4}" y="{y + 3:.1f}" text-anchor="end" font-size="7">{h % 24:02d}</text>')
        for mth in pd.date_range("2026-01-01", "2026-05-01", freq="MS"):
            x = cx + (mth - d0).days / ndays * cw
            out.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{cy}" y2="{cy + ch}" stroke="#ccc"/>')
            out.append(f'<text x="{x:.1f}" y="{cy + ch + 10}" font-size="7" text-anchor="middle">{mth.strftime("%b")}</text>')
        for r in q[q.pid == pid].itertuples():
            x = cx + ((r.local - d0).total_seconds() / 86400) / ndays * cw; y = cy + r.min_after4 / 1440 * ch
            col = "#d62728" if r.any_exclusion else ("#ff7f0e" if r.any_flag else "#777")
            out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="1.8" fill="{col}"/>')
        for r in f[f.pid == pid].itertuples():
            x = cx + ((r.local - d0).total_seconds() / 86400) / ndays * cw; y = cy + r.min_after4 / 1440 * ch
            st = 'stroke="#d62728" stroke-width="1"' if r.any_exclusion else ""
            out.append(f'<path d="M{x:.1f},{y - 2:.1f} l2,3.5 l-4,0 z" fill="#1f77b4" fill-opacity="0.6" {st}/>')
    out.append("</svg>")
    open(path, "w").write("\n".join(out))


if __name__ == "__main__":
    p, f = load_tests()
    q, excl = pvt_flags(p)
    fq = pef_flags(f, q)
    R = rider_summary(q, fq)
    keep = ["pid", "role", "timestamp", "node", "workday", "speed", "valid_taps", "total_taps", "percent_success", "false_starts",
            "n_anticip", "med_rt", "iqr_rt", "max_rt", "max_tap_gap_s", "dur_s", "P8_speed_minutes",
            "P1_incomplete", "P2_low_accuracy", "P3_false_starts", "P4_anticipations", "P5_slow_session", "P6_erratic",
            "P7_interruption", "P7b_long_rts_5_10s", "P8_moving", "P9_unusual_hour", "P10_near_duplicate",
            "P11_outside_enrolment", "any_exclusion", "any_flag"]
    q[keep].to_csv(os.path.join(FOUND, "qc_session_flags.csv"), index=False)
    fq[["pid", "role", "timestamp", "node", "workday", "peak_flow", "personal_best", "sitting", "sitting_size", "F1_low", "F2_high",
        "F3_poor_reproducibility", "F4_right_after_pvt", "F5_outside_enrolment", "any_exclusion"]].to_csv(os.path.join(FOUND, "qc_pef_flags.csv"), index=False)
    R.to_csv(os.path.join(FOUND, "qc_rider_summary.csv"), index=False)
    reg = register(R); reg.to_csv(os.path.join(FOUND, "outlier_register.csv"), index=False)
    timeline_svg(q, fq, os.path.join(FOUND, "qc_timeline.svg"))
    # ---- Task 5 features
    FP = features(q, "t_unix", "pvt"); FF = features(fq, "t_unix", "pef")
    fcols = ["clock_hour", "day_of_week", "hours_on_shift_before_test", "sensor_on_hours_before_test", "km_ridden_before_test",
             "rest_gap_hours", "tests_already_today", "days_since_enrolment"]
    FP[["pid", "role", "timestamp", "node", "workday", "shift_start_local", "test_before_shift", "gps_minutes_workday"] + fcols].to_csv(
        os.path.join(FOUND, "workday_features_pvt.csv"), index=False)
    FF[["pid", "role", "timestamp", "node", "workday", "shift_start_local", "test_before_shift", "gps_minutes_workday"] + fcols].to_csv(
        os.path.join(FOUND, "workday_features_pef.csv"), index=False)
    rid = FP[(FP.role == "rider") & ~FP.P11_outside_enrolment]
    C = rid[fcols].corr(); C.to_csv(os.path.join(FOUND, "feature_correlations.csv"), float_format="%.3f")
    # within-rider (demeaned) correlation for the key pair
    dm = rid[["pid", "clock_hour", "hours_on_shift_before_test", "sensor_on_hours_before_test"]].dropna()
    for c in ["clock_hour", "hours_on_shift_before_test", "sensor_on_hours_before_test"]:
        dm[c] = dm[c] - dm.groupby("pid")[c].transform("mean")
    print("within-rider r clock~hours_on_shift", round(dm.clock_hour.corr(dm.hours_on_shift_before_test), 3),
          "clock~sensor_on", round(dm.clock_hour.corr(dm.sensor_on_hours_before_test), 3))
    print(C.round(2).to_string())
    print("feature coverage (rider in-window PVT):", rid[fcols].notna().mean().round(3).to_dict())
    print("test_before_shift share", round(rid.test_before_shift.mean(), 3))
    print("PVT flag counts (riders):", q[q.role == "rider"][[c for c in keep if c.startswith("P") and c not in ("P8_speed_minutes", "P7b_long_rts_5_10s")] + ["any_exclusion", "any_flag"]].fillna(False).astype(bool).sum().to_dict())
    print("P8 evaluable", int(q[q.role == "rider"].P8_moving.notna().sum()))
    print("PEF flag counts (riders):", fq[fq.role == "rider"][["F1_low", "F2_high", "F3_poor_reproducibility", "F4_right_after_pvt", "F5_outside_enrolment", "any_exclusion"]].sum().to_dict())
    print("cohort median PVT/wk", R.attrs["cohort_median_pvt_per_week"])
    print(R[["pid", "enrol_weeks", "pvt_per_week", "share_days_with_test", "share_sessions_exclusion", "n_gaps_ge7d", "n_weeks_time_shift_ge4h", "share_tests_21_04", "practice_slope_log"] + [c for c in R.columns if c.startswith("R") and "_" in c and c[1].isdigit()]].to_string())
