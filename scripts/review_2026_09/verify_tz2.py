"""Timezone check: the stop-to-test movement dip around PVT times appears only if the naive
timestamps are read as Bangkok-local (compared with UTC and +/-1h shifts)."""
import _paths  # side effect: chdir into data/review_2026_09/
import numpy as np, pandas as pd
m = pd.read_csv("aq_minute.csv", usecols=["node", "minute", "disp_m"])
m["moving"] = m.disp_m.between(100, 2000)
mv = m.set_index(["node", "minute"]).moving
pv = pd.read_csv(f"{_paths.DATA}/pvt_aq_linked.csv", usecols=["timestamp", "node"]).dropna()
naive = pd.to_datetime(pv.timestamp)
offs = np.arange(-20, 21)
for label, shift_h in [("naive=Bangkok", 0), ("naive=UTC (i.e. +7h)", 7), ("naive=Bangkok-1h", -1), ("naive=Bangkok+1h", 1)]:
    unix = naive.dt.tz_localize("Asia/Bangkok").astype("int64") // 10**9 + shift_h * 3600
    acc = []
    for node, u in zip(pv.node, unix):
        try:
            s = mv.loc[node].reindex(((u // 60) * 60) + 60 * offs)
        except KeyError:
            continue
        if s.notna().sum() > 30:
            acc.append(s.to_numpy(dtype=float))
    A = 100 * np.nanmean(np.vstack(acc), axis=0)
    base = np.r_[A[:5], A[-5:]].mean()
    print(f"{label:24s} n={len(acc):4d} baseline(|off|>=16m)={base:5.1f}%  min over -5..+5 = {A[15:26].min():5.1f}%  "
          f"profile(-20..+20 step 4)={[round(x,1) for x in A[::4]]}")
