"""Build the linked PVT + air-quality-exposure dataset.

For every PVT session by a study rider, attach PM2.5 / temperature /
humidity exposure summaries over several candidate windows preceding the
test, plus the covariates the analysis plan calls for (practice/session
number, time of day, day of week, rider identity, approximate
time-on-shift).

Requires scripts/split_aq_by_node.py to have been run first (writes
data/aq_by_node/<node>.csv).

Output: data/pvt_aq_linked.csv
"""
import ast
import datetime as dt
import os

import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(__file__), "..")
AQ_BY_NODE_DIR = os.path.join(ROOT, "data", "aq_by_node")
OUT_PATH = os.path.join(ROOT, "data", "pvt_aq_linked.csv")

# Research-staff / test accounts identified in pvt.csv that are not real
# study riders (confirmed 2026-07-29) -- exclude from the cohort.
EXCLUDE_USERNAMES = {"Nussara", "Rach", "Rac", "test", "Preechai", "adisorn", "Patte"}

# Exposure windows to compute, in seconds, preceding each PVT test timestamp.
WINDOWS = {
    "1h": 3600,
    "4h": 4 * 3600,
}


def load_roster():
    df = pd.read_excel(
        os.path.join(ROOT, "2025-2026 Rider-Sensor-Summary.xlsx"), sheet_name="summary"
    )
    df["PVT user"] = df["PVT user"].astype(str).str.strip()
    df["Node"] = "Grab-" + df["Sensor no."].astype(str)
    return df[["PVT user", "Node", "Start date", "End date"]]


def resolve_node(username, test_date, roster):
    """Return the Node assigned to `username` covering `test_date`, or None.

    Handles sensors reassigned between riders over the study period (e.g.
    Grab-6021 covered Chaiyalit through 2026-02-26 then GAMPANAT from
    2026-03-04) by matching on the PVT user's active date range, not on
    sensor number alone.
    """
    rows = roster[roster["PVT user"] == username]
    for _, r in rows.iterrows():
        if r["Start date"].date() <= test_date <= r["End date"].date():
            return r["Node"]
    return None


def load_pvt():
    df = pd.read_csv(os.path.join(ROOT, "pvt.csv"))
    df["username"] = df["username"].astype(str).str.strip()
    df = df[~df["username"].isin(EXCLUDE_USERNAMES)].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    def resp_speed(s):
        try:
            vals = ast.literal_eval(s)
        except (ValueError, SyntaxError):
            return np.nan
        vals = [v for v in vals if v > 0]
        return np.nan if not vals else float(np.mean([1000.0 / v for v in vals]))

    def median_rt(s):
        try:
            vals = ast.literal_eval(s)
        except (ValueError, SyntaxError):
            return np.nan
        return np.nan if not vals else float(np.median(vals))

    df["response_speed_hz"] = df["response_times"].apply(resp_speed)
    df["median_rt_ms"] = df["response_times"].apply(median_rt)
    df["error_rate"] = 1 - df["percent_success"] / 100.0

    df = df.sort_values(["username", "timestamp"])
    df["session_number"] = df.groupby("username").cumcount() + 1
    df["hour_of_day"] = df["timestamp"].dt.hour + df["timestamp"].dt.minute / 60.0
    df["day_of_week"] = df["timestamp"].dt.day_name()
    df["test_date"] = df["timestamp"].dt.date
    return df


class NodeCache:
    """Loads and caches per-node AQ arrays for windowed lookups."""

    def __init__(self):
        self._cache = {}

    def get(self, node):
        if node not in self._cache:
            path = os.path.join(AQ_BY_NODE_DIR, f"{node}.csv")
            if not os.path.exists(path):
                self._cache[node] = None
                return None
            df = pd.read_csv(path, usecols=["Timestamp", "Relative humidity", "PM2.5", "Temperature"])
            df = df.sort_values("Timestamp")
            self._cache[node] = {
                "ts": df["Timestamp"].to_numpy(),
                "pm25": df["PM2.5"].to_numpy(dtype=float),
                "humidity": df["Relative humidity"].to_numpy(dtype=float),
                "temp": df["Temperature"].to_numpy(dtype=float),
            }
        return self._cache[node]

    def window_stats(self, node, ts_end, window_s):
        arrs = self.get(node)
        empty = {
            "pm25_mean": np.nan, "pm25_median": np.nan,
            "temp_mean": np.nan, "humidity_mean": np.nan, "n_obs": 0,
        }
        if arrs is None:
            return empty
        ts = arrs["ts"]
        lo = np.searchsorted(ts, ts_end - window_s, side="left")
        hi = np.searchsorted(ts, ts_end, side="right")
        if hi <= lo:
            return empty
        pm25 = arrs["pm25"][lo:hi]
        return {
            "pm25_mean": float(np.nanmean(pm25)),
            "pm25_median": float(np.nanmedian(pm25)),
            "temp_mean": float(np.nanmean(arrs["temp"][lo:hi])),
            "humidity_mean": float(np.nanmean(arrs["humidity"][lo:hi])),
            "n_obs": int(hi - lo),
        }

    def minutes_since_shift_start(self, node, ts_end, test_date):
        """Proxy for time-on-shift: minutes from that rider's first AQ
        reading on the test's calendar day to the test timestamp."""
        arrs = self.get(node)
        if arrs is None:
            return np.nan
        day_start = dt.datetime.combine(test_date, dt.time.min).timestamp()
        day_end = dt.datetime.combine(test_date, dt.time.max).timestamp()
        ts = arrs["ts"]
        lo = np.searchsorted(ts, day_start, side="left")
        hi = np.searchsorted(ts, min(day_end, ts_end), side="right")
        if hi <= lo:
            return np.nan
        return (ts_end - ts[lo]) / 60.0


def main():
    roster = load_roster()
    pvt = load_pvt()
    cache = NodeCache()

    records = []
    for row in pvt.itertuples(index=False):
        node = resolve_node(row.username, row.test_date, roster)
        rec = row._asdict()
        rec["node"] = node
        rec["timestamp_unix"] = row.timestamp.timestamp()
        if node is not None:
            for wname, wsec in WINDOWS.items():
                stats = cache.window_stats(node, rec["timestamp_unix"], wsec)
                for k, v in stats.items():
                    rec[f"{k}_{wname}"] = v
            rec["minutes_since_shift_start"] = cache.minutes_since_shift_start(
                node, rec["timestamp_unix"], row.test_date
            )
        else:
            for wname in WINDOWS:
                for k in ("pm25_mean", "pm25_median", "temp_mean", "humidity_mean", "n_obs"):
                    rec[f"{k}_{wname}"] = np.nan
            rec["minutes_since_shift_start"] = np.nan
        records.append(rec)

    out = pd.DataFrame.from_records(records)
    out.to_csv(OUT_PATH, index=False)

    n = len(out)
    covered = out["pm25_mean_1h"].notna().sum()
    print(f"{n} PVT sessions written to {OUT_PATH}")
    print(f"1h-window PM2.5 coverage: {covered}/{n} ({100 * covered / n:.1f}%)")


if __name__ == "__main__":
    main()
