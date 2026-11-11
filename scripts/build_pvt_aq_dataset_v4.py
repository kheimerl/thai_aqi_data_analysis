"""S1 Task 6: linked dataset v4 (independent of build_pvt_aq_dataset.py and analysis_health_v3.py).

Outputs: foundation/linked_v4_pvt.csv, foundation/linked_v4_pef.csv
One row per PVT session / peak-flow reading, all persons (riders R##, office O##); filter on `role`.
Clock: naive Bangkok local -> Unix via explicit -7 h. PVT timestamp = END of the 5-min test.
NO MODELS ARE FITTED HERE.

PM windows (valid PM 0..500, second-weighted; n_min = minutes with PM):
  pm_1h, pm_4h          [t0 - 1 h, t0), [t0 - 4 h, t0)       t0 = test start (PVT) / reading time (PEF)
  pm_sameday            workday start (04:00) -> t0            ("same-day mean so far")
  pm_sameday_cum        sum of minute PM / 60 over the same window (ug/m3 * h); = mean x hours of data
  pm_after              t1 -> end of workday (sensor-off)      t1 = test end (PVT) / reading time (PEF)  [placebo]
  pm_sameday_v3def      local midnight -> PVT timestamp (Task 1 / analysis_v3 definition, reproducibility)
  pm_regional_sameday   mean over the person's data hours (workday start -> t0) of the median hourly PM of the
                        OTHER rider nodes in the same city (>= 30 min PM each, >= 2 nodes); pm_local_sameday = pm_sameday - regional
  temp_sameday, rh_sameday  same window as pm_sameday
  v4.1 (2026-11-10, coordinator request; existing columns unchanged):
  pm_onshift(_nmin)     GPS shift start -> t0; NA if test_before_shift or no shift start
  pm_sameday_median, pm_1h_median   median of minute means in the window
  pm_sameday_clip, pm_1h_clip       second-weighted mean with minute means capped at the node's 99.9th pct (whole study)
  pm_sameday_peak5      max mean over 5-min clock blocks (>= 3 minutes with PM) in the same-day window
  cov_ok_*              1h >= 30 min, 4h >= 120, sameday >= 60, after >= 60, onshift >= 30 (minutes with PM)
  v4.2 (2026-11-10, S3 request; PVT file only; existing columns unchanged; needs >= 30 GPS minutes in the workday):
  pm_preshift(_nmin)       workday start 04:00 -> min(GPS shift start, t0); whole workday before t0 if no sustained movement
  pm_after_onshift(_nmin)  t1 -> last moving minute of the workday's last sustained-movement run (+1 min); NA if that is before t1
  pm_after_home(_nmin)     end of that run (or t1 if no movement after t1) -> end of workday
  last_move_local          local time of that last moving minute ('' if none)
  pm_calibrated_*       left empty: join calibration/pm_calibrated_hourly.csv (S2)
"""
import os, sys, ast
import numpy as np, pandas as pd
import warnings; warnings.filterwarnings("ignore", category=FutureWarning)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from foundation_common import *
from foundation_cohort_const import OFFICE_SENSOR

W = os.path.join(FOUND, "_work")


class Node:
    def __init__(self, node):
        d = pd.read_csv(os.path.join(W, f"minute_{node}.csv"), usecols=["minute_unix", "n_pm", "pm", "rh", "temp"])
        self.m = d.minute_unix.values
        self.pm = d.pm.values; self.npm = d.n_pm.values.astype(float)
        self.p999 = float(np.nanpercentile(self.pm, 99.9)) if np.isfinite(self.pm).any() else np.nan
        pmw = np.nan_to_num(d.pm.values * d.n_pm.values)
        self.c_pm = np.r_[0, np.cumsum(pmw)]; self.c_n = np.r_[0, np.cumsum(d.n_pm.values)]
        self.c_min = np.r_[0, np.cumsum(d.n_pm.values > 0)]; self.c_pmmin = np.r_[0, np.cumsum(np.nan_to_num(d.pm.values))]
        for k in ("rh", "temp"):
            v = d[k].values; setattr(self, f"c_{k}", np.r_[0, np.cumsum(np.nan_to_num(v))]); setattr(self, f"c_{k}n", np.r_[0, np.cumsum(~np.isnan(v))])

    def extra(self, u0, u1):
        """v4.1: median of minute means, clipped (node 99.9th pct) second-weighted mean, max 5-min block mean."""
        i, j = np.searchsorted(self.m, u0, "left"), np.searchsorted(self.m, u1, "left")
        pm = self.pm[i:j]; w = self.npm[i:j]; ok = ~np.isnan(pm) & (w > 0)
        if not ok.any():
            return dict(median=np.nan, clip=np.nan, peak5=np.nan)
        pm, w, mm = pm[ok], w[ok], self.m[i:j][ok]
        clip = float((np.minimum(pm, self.p999) * w).sum() / w.sum())
        blk = (mm // 300)
        df = pd.DataFrame({"b": blk, "pw": pm * w, "w": w, "one": 1}).groupby("b").sum()
        df = df[df.one >= 3]
        return dict(median=float(np.median(pm)), clip=clip, peak5=float((df.pw / df.w).max()) if len(df) else np.nan)

    def win(self, u0, u1):
        i, j = np.searchsorted(self.m, u0, "left"), np.searchsorted(self.m, u1, "left")
        n = self.c_n[j] - self.c_n[i]; nm = self.c_min[j] - self.c_min[i]
        out = dict(mean=(self.c_pm[j] - self.c_pm[i]) / n if n > 0 else np.nan, n_min=int(nm),
                   cum=(self.c_pmmin[j] - self.c_pmmin[i]) / 60 if nm > 0 else np.nan)
        for k in ("rh", "temp"):
            cn = getattr(self, f"c_{k}n"); c = getattr(self, f"c_{k}")
            out[k] = (c[j] - c[i]) / (cn[j] - cn[i]) if cn[j] - cn[i] > 0 else np.nan
        return out


def regional_table():
    H = pd.read_csv(os.path.join(FOUND, "pm_hourly_by_node.csv"),
                    usecols=["node", "node_class", "person", "hour_unix", "pm_mean", "minutes_pm"])
    C = pd.read_csv(os.path.join(FOUND, "cohort.csv")); city = dict(zip(C.pid, C.city))
    R = H[(H.node_class == "rider") & H.person.notna() & (H.person != "") & (H.minutes_pm >= 30)].copy()
    R["city"] = R.person.map(city)
    return R[["node", "city", "hour_unix", "pm_mean", "minutes_pm"]]


def regional(Rt, node, city, u0, u1):
    """Mean (weighted by own minutes) of the median of OTHER same-city rider nodes, over own data hours."""
    h0, h1 = (u0 // 3600) * 3600, u1
    sub = Rt[(Rt.city == city) & (Rt.hour_unix >= h0) & (Rt.hour_unix < h1)]
    own = sub[sub.node == node]
    oth = sub[sub.node != node].groupby("hour_unix").pm_mean.agg(["median", "size"])
    oth = oth[oth["size"] >= 2]
    j = own.set_index("hour_unix")[["minutes_pm"]].join(oth["median"], how="inner")
    return float((j["median"] * j.minutes_pm).sum() / j.minutes_pm.sum()) if len(j) else np.nan


def main():
    Q = pd.read_csv(os.path.join(FOUND, "qc_session_flags.csv")); FQ = pd.read_csv(os.path.join(FOUND, "qc_pef_flags.csv"))
    WP = pd.read_csv(os.path.join(FOUND, "workday_features_pvt.csv")); WF = pd.read_csv(os.path.join(FOUND, "workday_features_pef.csv"))
    C = pd.read_csv(os.path.join(FOUND, "cohort.csv"))
    key = pd.read_csv(os.path.join(FOUND, "_private_key.csv")); user_of = dict(zip(key.pid, key.user))
    pvt = pd.read_csv(os.path.join(DATA, "pvt.csv")); pvt["user"] = pvt.username.str.strip()
    pvt = pvt[pvt.user.isin(set(key.user))].copy(); pvt["pid"] = pvt.user.map(dict(zip(key.user, key.pid)))
    Rt = regional_table(); cache = {}
    city = dict(zip(C.pid, C.city)); role = dict(zip(C.pid, C.role))

    def nodeobj(n):
        if n not in cache: cache[n] = Node(n)
        return cache[n]

    # ------------------------------ PVT outcomes
    pvt = pvt.merge(Q[["pid", "timestamp", "node", "workday"]], on=["pid", "timestamp"], how="left")
    rts = pvt.response_times.map(lambda s: np.asarray(ast.literal_eval(s), float))
    pvt["y1_speed_rtgt0"] = rts.map(lambda r: np.mean(1000 / r[r > 0]) if (r > 0).any() else np.nan)
    pvt["y1_speed_rt100"] = rts.map(lambda r: np.mean(1000 / r[r >= 100]) if (r >= 100).any() else np.nan)
    own_med = {p: np.median(np.concatenate([r[r >= 100] for r in rts[pvt.pid == p]])) for p in pvt.pid.unique()}
    pvt["own_median_rt"] = pvt.pid.map(own_med)
    def lapse(r, thr):
        r = r[r >= 100]; return 100 * np.mean(r >= thr) if len(r) else np.nan
    pvt["y2_lapse_pct_2xmed"] = [lapse(r, 2 * m) for r, m in zip(rts, pvt.own_median_rt)]
    pvt["y2_lapse_pct_500"] = [lapse(r, 500) for r in rts]
    pvt["y2_lapse_pct_1000"] = [lapse(r, 1000) for r in rts]
    def slow10(r):
        r = np.sort(r[r >= 100])[::-1]; k = max(1, int(np.ceil(0.1 * len(r)))) if len(r) else 0
        return np.mean(1000 / r[:k]) if k else np.nan
    pvt["y3_slowest10_speed"] = rts.map(slow10)
    pvt["n_trials_rt100"] = rts.map(lambda r: int((r >= 100).sum()))
    pvt["local_end"] = pd.to_datetime(pvt.timestamp)
    pvt["t1"] = local_to_unix(pvt.local_end); pvt["t0"] = pvt.t1 - 300
    pvt["session_number"] = pvt.sort_values("local_end").groupby("pid").cumcount() + 1

    def add_v41(df, t0col):
        """v4.1 (2026-11-10): on-shift mean (GPS shift start -> t0) and coverage flags."""
        ss = pd.to_datetime(df.shift_start_local.replace("", np.nan), errors="coerce")
        ssu = local_to_unix(ss)
        mean, nmin = [], []
        for r, u in zip(df.itertuples(), ssu):
            if pd.isna(r.node) or pd.isna(u) or bool(r.test_before_shift):
                mean.append(np.nan); nmin.append(0); continue
            w = nodeobj(int(r.node)).win(u, getattr(r, t0col)); mean.append(w["mean"]); nmin.append(w["n_min"])
        df["pm_onshift"] = mean; df["pm_onshift_nmin"] = nmin
        df["cov_ok_1h"] = df.pm_1h_nmin.fillna(0) >= 30
        df["cov_ok_4h"] = df.pm_4h_nmin.fillna(0) >= 120
        df["cov_ok_sameday"] = df.pm_sameday_nmin.fillna(0) >= 60
        df["cov_ok_after"] = df.pm_after_nmin.fillna(0) >= 60
        df["cov_ok_onshift"] = df.pm_onshift_nmin.fillna(0) >= 30
        return df

    def add_v42(df):
        """v4.2 (2026-11-10, S3 request): home vs on-road split of the workday (PVT only).
        Needs a GPS workday (>= 30 GPS minutes). Shift start / last move = first / last minute of a sustained-movement
        run (QC_RULES D). If the GPS workday has no sustained movement, the whole workday counts as non-riding."""
        from foundation_qc_features import workday_info
        cols = {k: [] for k in ["pm_preshift", "pm_preshift_nmin", "pm_after_onshift", "pm_after_onshift_nmin",
                                "pm_after_home", "pm_after_home_nmin", "last_move_local"]}
        for r in df.itertuples():
            vals = dict(pm_preshift=np.nan, pm_preshift_nmin=0, pm_after_onshift=np.nan, pm_after_onshift_nmin=0,
                        pm_after_home=np.nan, pm_after_home_nmin=0, last_move_local="")
            if not pd.isna(r.node):
                wd = pd.Timestamp(r.workday); wi = workday_info(int(r.node), wd)
                if wi["gps_minutes"] >= 30:
                    n = nodeobj(int(r.node)); wd0 = local_to_unix(wd) + 4 * 3600; wd1 = wd0 + 86400
                    ss, lm = wi["shift_start"], wi["last_activity"]
                    w = n.win(wd0, r.t0 if np.isnan(ss) else min(ss, r.t0))
                    vals.update(pm_preshift=w["mean"], pm_preshift_nmin=w["n_min"])
                    home0 = r.t1
                    if not np.isnan(lm):
                        vals["last_move_local"] = str(unix_to_local([lm])[0])[:16]
                        if lm >= r.t1:
                            w = n.win(r.t1, lm + 60); vals.update(pm_after_onshift=w["mean"], pm_after_onshift_nmin=w["n_min"])
                            home0 = lm + 60
                    w = n.win(home0, wd1); vals.update(pm_after_home=w["mean"], pm_after_home_nmin=w["n_min"])
            for k in cols: cols[k].append(vals[k])
        for k, v in cols.items(): df[k] = v
        return df

    def add_pm(df, t0col, t1col, is_pvt):
        rows = []
        for r in df.itertuples():
            o = {}
            if pd.isna(r.node):
                rows.append(o); continue
            n = nodeobj(int(r.node)); t0, t1 = getattr(r, t0col), getattr(r, t1col)
            wd0 = local_to_unix(pd.Timestamp(r.workday)) + 4 * 3600; wd1 = wd0 + 86400
            for lab, a, b in (("pm_1h", t0 - 3600, t0), ("pm_4h", t0 - 4 * 3600, t0)):
                w = n.win(a, b); o[lab] = w["mean"]; o[lab + "_nmin"] = w["n_min"]
            w = n.win(wd0, t0)
            o.update(pm_sameday=w["mean"], pm_sameday_nmin=w["n_min"], pm_sameday_cum=w["cum"], temp_sameday=w["temp"], rh_sameday=w["rh"])
            e = n.extra(wd0, t0); o.update(pm_sameday_median=e["median"], pm_sameday_clip=e["clip"], pm_sameday_peak5=e["peak5"])
            e = n.extra(t0 - 3600, t0); o.update(pm_1h_median=e["median"], pm_1h_clip=e["clip"])
            w = n.win(t1, wd1); o.update(pm_after=w["mean"], pm_after_nmin=w["n_min"])
            if is_pvt:
                ld = (pd.Timestamp(r.local_end).normalize() - EPOCH).total_seconds() - BKK_OFFSET_S
                w = n.win(ld, t1); o.update(pm_sameday_v3def=w["mean"], pm_sameday_v3def_nmin=w["n_min"])
            reg = regional(Rt, int(r.node), city[r.pid], wd0, t0) if role[r.pid] == "rider" else np.nan
            o["pm_regional_sameday"] = reg
            o["pm_local_sameday"] = o["pm_sameday"] - reg if not np.isnan(reg) and not np.isnan(o["pm_sameday"]) else np.nan
            rows.append(o)
        P = pd.DataFrame(rows, index=df.index)
        for c in ["pm_calibrated_1h", "pm_calibrated_4h", "pm_calibrated_sameday", "pm_calibrated_after"]:
            P[c] = np.nan
        return df.join(P)

    pvt = add_pm(pvt, "t0", "t1", True)
    pvt["city"] = pvt.pid.map(city); pvt["role"] = pvt.pid.map(role)
    pvt = pvt.merge(Q.drop(columns=["role", "node", "workday", "speed", "valid_taps", "total_taps", "percent_success"]), on=["pid", "timestamp"], how="left")
    pvt = pvt.merge(WP.drop(columns=["role", "node", "workday"]), on=["pid", "timestamp"], how="left")
    pvt = add_v41(pvt, "t0")
    pvt = add_v42(pvt)
    hand = dict(zip(C.pid, C.handover)); pvt["sensor_handover_person"] = pvt.pid.map(hand)
    pvt["late_starter_handover"] = pvt.pid.isin(["R12", "R14"])
    drop = ["username", "user", "response_times", "stimulation_times", "all_tap_times", "local_end"]
    out = pvt.drop(columns=drop).rename(columns={"timestamp": "timestamp_end_local", "t0": "test_start_unix", "t1": "test_end_unix"})
    out = out.sort_values(["pid", "test_end_unix"])
    for c in ("test_end_unix", "test_start_unix"):   # S7 audit S1-1: exact integer seconds, not %.6g
        out[c] = out[c].round().astype("int64")
    out.to_csv(os.path.join(FOUND, "linked_v4_pvt.csv"), index=False, float_format="%.6g")

    # ------------------------------ peak flow
    pef = pd.read_csv(os.path.join(DATA, "peakflow.csv")); pef["user"] = pef.username.str.strip()
    pef = pef[pef.user.isin(set(key.user))].copy(); pef["pid"] = pef.user.map(dict(zip(key.user, key.pid)))
    FQ = FQ.drop_duplicates(["pid", "timestamp"]); WF = WF.drop_duplicates(["pid", "timestamp"])
    pef["exact_duplicate"] = pef.duplicated(["pid", "timestamp", "peak_flow"])  # 2nd copy of an identical row (R02 x1, O01 x1)
    pef = pef.merge(FQ[["pid", "timestamp", "node", "workday", "personal_best"]], on=["pid", "timestamp"], how="left")
    pef["local"] = pd.to_datetime(pef.timestamp); pef["t"] = local_to_unix(pef.local)
    pef["pct_pb"] = 100 * pef.peak_flow / pef.personal_best
    h = pef.local.dt.hour
    pef["period"] = np.where((h >= 4) & (h <= 13), "morning", np.where((h >= 15) | (h < 4), "evening", "midday"))
    pef = pef.sort_values(["pid", "t"])
    g = pef.groupby(["pid", "workday", "period"]).t
    pef["first_morning_of_workday"] = (pef.period == "morning") & (pef.t == g.transform("min"))
    pef["last_evening_of_workday"] = (pef.period == "evening") & (pef.t == g.transform("max"))
    pef = add_pm(pef, "t", "t", False)
    pef["city"] = pef.pid.map(city); pef["role"] = pef.pid.map(role)
    pef = pef.merge(FQ.drop(columns=["role", "node", "workday", "peak_flow", "personal_best"]), on=["pid", "timestamp"], how="left")
    pef = pef.merge(WF.drop(columns=["role", "node", "workday"]), on=["pid", "timestamp"], how="left")
    pef = add_v41(pef, "t")
    pef["sensor_handover_person"] = pef.pid.map(hand); pef["late_starter_handover"] = pef.pid.isin(["R12", "R14"])
    outf = pef.drop(columns=["username", "user", "local"]).rename(columns={"timestamp": "timestamp_local", "t": "reading_unix"})
    outf["reading_unix"] = outf["reading_unix"].round().astype("int64")   # S7 audit S1-1
    outf.to_csv(os.path.join(FOUND, "linked_v4_pef.csv"), index=False, float_format="%.6g")
    print("pvt rows", len(out), "riders", (out.role == "rider").sum(), "| pef rows", len(outf), "riders", (outf.role == "rider").sum())


if __name__ == "__main__":
    main()
