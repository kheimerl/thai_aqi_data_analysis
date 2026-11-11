"""v4.1 item 5: is the onboard humidity/temperature ambient? Exposure-free, outcome-free.
Uses minute files; 'outdoors' = moving minutes (GPS speed > 5 km/h, rider nodes, assigned wearer).
Dewpoint (Magnus): Td = b*g/(a-g), g = ln(RH/100) + a*T/(b+T), a=17.62, b=243.12."""
import os, sys, glob
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from foundation_common import *
nodes = pd.read_csv(os.path.join(FOUND, "nodes.csv"))
city = dict(zip(nodes.node, nodes.city)); cls = dict(zip(nodes.node, nodes["class"]))
rows = []
for f in glob.glob(os.path.join(FOUND, "_work", "minute_*.csv")):
    n = int(os.path.basename(f)[7:-4])
    d = pd.read_csv(f, usecols=["minute_unix", "rh", "temp", "speed_kmh"])
    d["node"] = n; rows.append(d)
D = pd.concat(rows); D["city"] = D.node.map(city); D["class"] = D.node.map(cls)
loc = unix_to_local(D.minute_unix); D["hour"] = loc.hour; D["month"] = loc.month
a, b = 17.62, 243.12
g = np.log(D.rh.clip(lower=1) / 100) + a * D.temp / (b + D.temp); D["dewpt"] = b * g / (a - g)
print("ALL minutes: RH pct", np.nanpercentile(D.rh, [1, 50, 99, 99.9]).round(1), "max", round(D.rh.max(), 1),
      "| share minutes RH>85%:", round((D.rh > 85).mean(), 5), "| T pct", np.nanpercentile(D.temp, [1, 50, 99]).round(1))
R = D[D["class"] == "rider"]
mv = R[R.speed_kmh > 5]
print("rider moving minutes", len(mv))
t = mv.groupby(["city", pd.cut(mv.hour, [0, 6, 9, 12, 15, 18, 21, 24], right=False)]).agg(
    n=("rh", "size"), T_med=("temp", "median"), RH_med=("rh", "median"), RH_p95=("rh", lambda x: np.nanpercentile(x, 95)),
    Td_med=("dewpt", "median")).round(1)
print(t.to_string())
m = mv[(mv.hour >= 6) & (mv.hour < 9)].groupby(["city", "month"]).agg(n=("rh", "size"), T_med=("temp", "median"), RH_med=("rh", "median"), Td_med=("dewpt", "median")).round(1)
print("\nmorning 06-09 moving, by month\n", m.to_string())
st = R[R.speed_kmh < 2]
print("\nstationary vs moving medians (riders): T", round(st.temp.median(), 1), "vs", round(mv.temp.median(), 1),
      "| RH", round(st.rh.median(), 1), "vs", round(mv.rh.median(), 1))
# warm-up: does temperature fall over the first minutes after sensor-on? (minutes since first data of the day)
R = R.sort_values(["node", "minute_unix"]); R["day"] = (R.minute_unix + 7 * 3600) // 86400
R["k"] = R.groupby(["node", "day"]).cumcount()
print("\nT and RH by minutes since day's first data (riders):")
print(R[R.k.isin([0, 5, 15, 30, 60, 120])].groupby("k")[["temp", "rh"]].median().round(1).T.to_string())
