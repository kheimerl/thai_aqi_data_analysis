"""S1 Task 2: roster checks, cohort.csv, nodes.csv, office-worker table (confirmed by user 2026-11-09; O05/6006 dropped 2026-11-10)."""
import os, sys, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from foundation_common import *
from foundation_ids import person_table, assignments
from foundation_task1_reproduce import roster

W = os.path.join(FOUND, "_work")
# Proposed office sensor mapping (6006 -> Saiyud is PROBABLE, awaiting user confirmation)
from foundation_cohort_const import OFFICE_SENSOR, DROPPED_NODES, DROPPED_PIDS
OTHER_NODES = {6008: "other (stationary, co-located with BKK office sensors; owner unknown)"}

H = pd.read_csv(os.path.join(FOUND, "pm_hourly_by_node.csv")); H["date"] = H.hour_local.str[:10]
pt = person_table(); asg = assignments(); m = read_mapping(); ro = roster()
pvt = pd.read_csv(os.path.join(DATA, "pvt.csv")); pef = pd.read_csv(os.path.join(DATA, "peakflow.csv"))
for d in (pvt, pef):
    d["user"] = d.username.str.strip(); d["date"] = d.timestamp.str[:10]

# ---- (a) mapping vs roster
chk = {}
mm = m[["user", "sensor", "city"]].copy()
rr = ro.groupby("user").sensor.apply(lambda s: sorted(set(s))).to_dict()
chk["same_people"] = set(mm.user) == set(ro.user)
chk["sensor_mismatch_mapping_vs_roster"] = {pt.set_index("user").pid[u]: {"mapping": int(s), "roster": [int(x) for x in rr[u]]}
                                            for u, s in zip(mm.user, mm.sensor) if [s] != rr[u]}
gaps = []
for sensor, g in asg.sort_values("start").groupby("sensor"):
    pass
def gap_info(sensor, d0, d1, label):
    h = H[(H.node == sensor) & (H.date >= d0) & (H.date <= d1)]
    return dict(label=label, sensor=sensor, gap_from=d0, gap_to=d1, data_days=int(h.date.nunique()), data_hours=int(len(h)),
                pvt_sessions=0, note="")
a = asg.copy()
gaps.append(gap_info(6021, "2026-02-27", "2026-03-03", "6021 handover R11->R12"))
gaps.append(gap_info(6022, "2026-03-26", "2026-03-26", "6022 handover R13->R14"))
gaps.append(gap_info(6029, "2026-01-29", "2026-02-12", "R22 6029 after end"))
gaps.append(gap_info(6032, "2026-01-29", "2026-02-12", "R22 6032 before start"))
gaps.append(gap_info(6017, "2026-02-27", "2026-05-01", "6017 after R07 returned equipment"))
# overlaps between consecutive users of a sensor
ov = []
for s, g in a.groupby("sensor"):
    g = g.sort_values("start")
    for i in range(1, len(g)):
        ov.append(dict(sensor=s, prev=g.iloc[i-1].pid, nxt=g.iloc[i].pid, prev_end=str(g.iloc[i-1].end.date()),
                       next_start=str(g.iloc[i].start.date()), overlap=bool(g.iloc[i].start <= g.iloc[i-1].end)))
chk["handover_overlap"] = ov
chk["gaps"] = gaps

# ---- cohort.csv
rows = []
for _, p in pt.iterrows():
    if p.role == "rider":
        sens = asg[asg.pid == p.pid]
        sensors = "/".join(str(int(s)) for s in sens.sensor)
        days = 0
        for _, s in sens.iterrows():
            h = H[(H.node == s.sensor) & (H.date >= str(s.start.date())) & (H.date <= str(s.end.date()))]
            days += h.date.nunique()
        handover = bool(sens.sensor.isin([6021, 6022]).any())
        city = p.city
    else:
        sensor, city = OFFICE_SENSOR[p.user]; sensors = str(sensor)
        days = H[H.node == sensor].date.nunique(); handover = False
    tp = pvt[pvt.user == p.user]; tf = pef[pef.user == p.user]
    dates = pd.concat([tp.date, tf.date])
    rows.append(dict(pid=p.pid, role=p.role, city=city, sensor=sensors, handover=handover,
                     sensor_switch=(p.role == "rider" and len(sensors.split("/")) > 1 and not handover),
                     enrol_start=str(p.start.date()) if p.role == "rider" else "", enrol_end=str(p.end.date()) if p.role == "rider" else "",
                     n_pvt=len(tp), n_pef=len(tf), sensor_days=int(days),
                     first_test=dates.min() if len(dates) else "", last_test=dates.max() if len(dates) else "",
                     office_sensor_status=("" if p.role == "rider" else ("probable, unconfirmed" if p.user == "Saiyud" else
                                           "confirmed by officeworker.txt" if p.user != "Saiyud" else ""))))
C = pd.DataFrame(rows)
C["office_sensor_status"] = [DROPPED_PIDS.get(p, st) for p, st in zip(C.pid, C.office_sensor_status)]  # S7 audit S1-3
# 2026-11-11 (coordinator): design had 20 rider slots; R12 and R14 took over CMI slots 6021/6022 mid-study.
# True = original slot holder, False = replacement rider; blank for office workers (no slots).
C["slot_original"] = [(p not in ("R12", "R14")) if r == "rider" else None for p, r in zip(C.pid, C.role)]

# matched city-hours: office node hour (>=30 min PM) with >=1 rider node in same city same hour (>=30 min PM)
R = H[(H.node_class == "rider") & (H.person != "") & (H.minutes_pm >= 30)].copy()
R["city"] = R.person.map(dict(zip(C.pid, C.city)))
rider_hours = R.groupby(["city", "hour_local"]).size()
mh = {}
for u, (sensor, default_city) in OFFICE_SENSOR.items():
    o = H[(H.node == sensor) & (H.minutes_pm >= 30)].copy()
    o["city"] = np.where(o.lat_median.notna(), np.where(o.lat_median > 16, "CMI", "BKK"), default_city)
    key = list(zip(o.city, o.hour_local))
    hit = [k in rider_hours.index for k in key]
    mh[u] = dict(hours=int(sum(hit)), dates=int(o[hit].date.nunique()) if any(hit) else 0,
                 hours_city_by_gps=int(o.lat_median.notna().sum()), hours_in_cmi_by_gps=int((o.lat_median > 16).sum()))
pid_of = dict(zip(pt.user, pt.pid))
C["matched_city_hours"] = C.pid.map({pid_of[u]: v["hours"] for u, v in mh.items()})
C["matched_dates"] = C.pid.map({pid_of[u]: v["dates"] for u, v in mh.items()})
C.to_csv(os.path.join(FOUND, "cohort.csv"), index=False)

# ---- nodes.csv (no coordinates)
I = pd.read_csv(os.path.join(W, "node_inventory_internal.csv"))
sensor_owner = {}
for _, s in asg.iterrows():
    sensor_owner.setdefault(int(s.sensor), []).append(s.pid)
for u, (s, c) in OFFICE_SENSOR.items():
    sensor_owner.setdefault(s, []).append(pid_of[u])
I["class"] = I.node.map(lambda n: "rider" if n in set(asg.sensor) else ("office" if n in [v[0] for v in OFFICE_SENSOR.values()] else "other"))
I["person"] = I.node.map(lambda n: "/".join(sensor_owner.get(n, [])))
I["status"] = I.node.map(lambda n: DROPPED_NODES[n] + " (was: probable owner O05)" if n in DROPPED_NODES else ("unassigned; stationary-looking" if n == 6008 else ""))
I["class"] = [("dropped" if n in DROPPED_NODES else c) for n, c in zip(I.node, I["class"])]
N = I[["node", "class", "person", "city_gps", "first_local", "last_local", "days_data", "hours_data", "hours_pm_ge30min",
       "share_hours_gps", "share_hours_moving", "share_gps_hours_cmi", "status"]].rename(columns={"city_gps": "city"})
N.to_csv(os.path.join(FOUND, "nodes.csv"), index=False)

json.dump(dict(checks=chk, office_matched=mh), open(os.path.join(W, "task2_checks.json"), "w"), indent=1, default=str)
print(json.dumps(chk, indent=1, default=str)); print(json.dumps(mh, indent=1))
print(C.to_string()); print(N.to_string())
