"""S1: minute GPS speed -> hourly per-node file (for S2) + node inventory.

Outputs
  foundation/_work/minute_<node>.csv  (minute table incl. speed; internal)
  foundation/pm_hourly_by_node.csv    (node x hour; contains median GPS -> internal to S2/S3/S4, never publish)
  foundation/_work/node_inventory_internal.csv (with median positions; internal)
Rules
  * minute PM = mean of valid seconds (0..500); minute position = mean of valid GPS fixes
  * minute speed (km/h) = haversine(prev GPS minute, this GPS minute) / dt, only if dt <= 3 min
  * moving minute: speed > 5 km/h; stationary minute: speed < 2 km/h
  * moving hour: >= 10 GPS-speed minutes and median minute speed > 5 km/h... (share_hours_moving)
"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from foundation_common import *
from foundation_ids import assignments, person_table

W = os.path.join(FOUND, "_work"); AQM = os.path.join(W, "aq_min")
RIDER_NODES = None


def haversine_km(lat1, lon1, lat2, lon2):
    p = np.pi / 180
    a = np.sin((lat2 - lat1) * p / 2) ** 2 + np.cos(lat1 * p) * np.cos(lat2 * p) * np.sin((lon2 - lon1) * p / 2) ** 2
    return 12742 * np.arcsin(np.sqrt(a))


def minute_table(node):
    d = pd.read_csv(os.path.join(AQM, f"{node}.csv"))
    d["pm"] = np.where(d.n_pm > 0, d.pm_sum / d.n_pm.clip(lower=1), np.nan)
    d["rh"] = np.where(d.n_rh > 0, d.rh_sum / d.n_rh.clip(lower=1), np.nan)
    d["temp"] = np.where(d.n_t > 0, d.t_sum / d.n_t.clip(lower=1), np.nan)
    d["lat"] = np.where(d.n_gps > 0, d.lat_sum / d.n_gps.clip(lower=1), np.nan)
    d["lon"] = np.where(d.n_gps > 0, d.lon_sum / d.n_gps.clip(lower=1), np.nan)
    g = d[d.lat.notna()]
    dt = g.minute_unix.diff() / 3600
    km = haversine_km(g.lat.shift(), g.lon.shift(), g.lat, g.lon)
    spd = (km / dt).where(dt * 60 <= 3)
    d["speed_kmh"] = np.nan; d.loc[g.index, "speed_kmh"] = spd.values
    d["km_step"] = np.nan; d.loc[g.index, "km_step"] = km.where(dt * 60 <= 3).values
    # drop GPS jumps: > 150 km/h is not riding
    bad = d.speed_kmh > 150
    d.loc[bad, ["speed_kmh", "km_step"]] = np.nan
    d["node"] = node
    return d[["node", "minute_unix", "n_rows", "n_pm", "pm", "rh", "temp", "n_gps", "lat", "lon", "speed_kmh", "km_step"]]


def main():
    asg = assignments()
    pt = person_table()
    rider_sensors = set(asg.sensor)
    # S7 audit S1-2: office classes and persons from the shared, user-confirmed table (was hard-coded 6009/6010/6033)
    from foundation_cohort_const import OFFICE_SENSOR, DROPPED_NODES
    pid_of = dict(zip(pt.user, pt.pid))
    office_nodes = {s: "office" for s, _ in OFFICE_SENSOR.values()}
    office_person = {s: pid_of[u] for u, (s, _) in OFFICE_SENSOR.items()}
    for n in DROPPED_NODES: office_nodes[n] = "dropped"; office_person.pop(n, None)
    hourly, inv = [], []
    for f in sorted(os.listdir(AQM)):
        node = int(f.split(".")[0])
        d = minute_table(node)
        d.to_csv(os.path.join(W, f"minute_{node}.csv"), index=False)
        d["hour_unix"] = (d.minute_unix // 3600) * 3600
        h = d.groupby("hour_unix").agg(
            minutes_data=("n_rows", lambda x: int((x > 0).sum())), minutes_pm=("n_pm", lambda x: int((x > 0).sum())),
            n_sec_pm=("n_pm", "sum"), pm_mean=("pm", "mean"), rh_mean=("rh", "mean"), temp_mean=("temp", "mean"),
            minutes_gps=("n_gps", lambda x: int((x > 0).sum())), lat_median=("lat", "median"), lon_median=("lon", "median"),
            minutes_speed=("speed_kmh", "count"), minutes_moving_gt5=("speed_kmh", lambda x: int((x > 5).sum())),
            minutes_stationary_lt2=("speed_kmh", lambda x: int((x < 2).sum())), km=("km_step", "sum"),
            speed_median=("speed_kmh", "median")).reset_index()
        # second-weighted hourly PM (minute means weighted by valid seconds)
        w = d.assign(pmw=d.pm * d.n_pm).groupby("hour_unix").pmw.sum()
        h["pm_mean"] = (w.values / h.n_sec_pm.replace(0, np.nan)).values
        h.insert(0, "node", node)
        h["hour_local"] = unix_to_local(h.hour_unix).strftime("%Y-%m-%d %H:00")
        # person for this hour (riders by roster date range)
        h["date_local"] = pd.to_datetime(h.hour_local.str[:10])
        h["person"] = ""
        for _, a in asg[asg.sensor == node].iterrows():
            sel = (h.date_local >= a.start) & (h.date_local <= a.end)
            h.loc[sel, "person"] = a.pid
        cls = "rider" if node in rider_sensors else office_nodes.get(node, "other")
        h["node_class"] = cls
        if cls == "office": h["person"] = office_person[node]
        h["pm_calibrated"] = np.nan
        hourly.append(h.drop(columns="date_local"))
        moving_hour = (h.minutes_speed >= 10) & (h.speed_median > 5)
        lat_m, lon_m = d.lat.median(), d.lon.median()
        inv.append(dict(node=node, node_class=cls,
                        first_local=str(unix_to_local([d.minute_unix.min()])[0]), last_local=str(unix_to_local([d.minute_unix.max()])[0]),
                        hours_data=int((h.minutes_data > 0).sum()), hours_pm_ge30min=int((h.minutes_pm >= 30).sum()),
                        days_data=int(h.hour_local.str[:10].nunique()),
                        share_hours_gps=round(float((h.minutes_gps > 0).mean()), 3),
                        hours_with_speed=int((h.minutes_speed >= 10).sum()),
                        share_hours_moving=round(float(moving_hour.sum() / max(1, (h.minutes_speed >= 10).sum())), 3),
                        lat_median=lat_m, lon_median=lon_m,
                        city_gps=("CMI" if lat_m > 16 else "BKK") if not np.isnan(lat_m) else "no GPS",
                        share_gps_hours_cmi=round(float((h.lat_median > 16).sum() / max(1, h.lat_median.notna().sum())), 3),
                        pm_median_hourly=round(float(h.pm_mean.median()), 1)))
    H = pd.concat(hourly, ignore_index=True)
    cols = ["node", "node_class", "person", "hour_unix", "hour_local", "pm_mean", "pm_calibrated", "rh_mean", "temp_mean",
            "minutes_data", "minutes_pm", "n_sec_pm", "minutes_gps", "lat_median", "lon_median", "minutes_speed",
            "minutes_moving_gt5", "minutes_stationary_lt2", "speed_median", "km"]
    H[cols].to_csv(os.path.join(FOUND, "pm_hourly_by_node.csv"), index=False, float_format="%.4f")
    I = pd.DataFrame(inv)
    I.to_csv(os.path.join(W, "node_inventory_internal.csv"), index=False)
    print(I.drop(columns=["lat_median", "lon_median"]).to_string())
    print("hourly rows", len(H))


if __name__ == "__main__":
    main()
