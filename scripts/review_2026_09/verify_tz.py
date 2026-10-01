"""Is the naive PVT timestamp Bangkok-local or UTC? Compare against sensor movement."""
import _paths  # side effect: chdir into data/review_2026_09/
import numpy as np, pandas as pd
m = pd.read_csv("aq_minute.csv")
m["moving"] = m.disp_m.between(100, 2000)
t = pd.to_datetime(m.minute, unit="s", utc=True).dt.tz_convert("Asia/Bangkok")
m["hr"] = t.dt.hour
print("Diurnal moving fraction by Bangkok hour (study-rider nodes):")
prof = m[m.node.str[-4:].astype(int).between(6011, 6032)].groupby("hr").moving.mean()
print((100 * prof).round(1).to_dict())

pv = pd.read_csv(f"{_paths.DATA}/pvt_aq_linked.csv", usecols=["timestamp", "username", "node"])
pv = pv.dropna(subset=["node"])
naive = pd.to_datetime(pv.timestamp)
res = {}
mv = m.set_index(["node", "minute"]).moving
for label, tz in [("naive=Bangkok", "Asia/Bangkok"), ("naive=UTC", "UTC")]:
    unix = naive.dt.tz_localize(tz).astype("int64") // 10**9
    frac_near, stat_at = [], []
    for node, u in zip(pv.node, unix):
        mins = ((u // 60) * 60) + 60 * np.arange(-60, 61)
        try:
            s = mv.loc[node].reindex(mins)
        except KeyError:
            continue
        if s.notna().sum() < 60:
            continue
        frac_near.append(s.mean())
        # during the ~5-min test itself, is the sensor stationary?
        stat_at.append(1 - s.iloc[60:66].mean())
    res[label] = (np.nanmean(frac_near), np.nanmean(stat_at), len(frac_near))
    print(f"{label}: sessions w/ data={len(frac_near)}  moving frac within +-1h={np.nanmean(frac_near):.3f}  "
          f"stationary frac during test minutes={np.nanmean(stat_at):.3f}")

# Event-locked profile: moving fraction at offsets around the test (Bangkok hypothesis)
unix = naive.dt.tz_localize("Asia/Bangkok").astype("int64") // 10**9
offs = np.arange(-30, 31)
acc = []
for node, u in zip(pv.node, unix):
    try:
        s = mv.loc[node].reindex(((u // 60) * 60) + 60 * offs)
    except KeyError:
        continue
    if s.notna().sum() > 40:
        acc.append(s.to_numpy(dtype=float))
A = np.nanmean(np.vstack(acc), axis=0)
print("Event-locked moving fraction (Bangkok hypothesis), minute offsets -30..+30 step 5:")
print({int(o): round(100 * a, 1) for o, a in zip(offs[::5], A[::5])})
