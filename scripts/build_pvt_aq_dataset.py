"""Build the linked PVT + air-quality-exposure dataset.

For every PVT session by a study rider, attach PM2.5 / temperature /
humidity exposure summaries plus the covariates the analysis plan calls
for (practice/session number, time of day, day of week, rider identity,
shift duration).

Primary exposure window (PLAN.md #3): cumulative same-day, from that
rider's first AQ reading that calendar day (shift start) to the test
timestamp. Secondary/sensitivity windows: 1h and 4h trailing.

Requires scripts/split_aq_by_node.sh to have been run first (writes
data/aq_by_node/<node>.csv).

Output: data/pvt_aq_linked.csv
"""
import ast
import datetime as dt
import os
import warnings

import numpy as np
import pandas as pd

# Some AQ rows have blank temperature/humidity fields, so narrow windows
# occasionally contain no valid readings for one field even though PM2.5
# is present -- nanmean/nanmedian correctly return NaN for these, but warn.
warnings.filterwarnings("ignore", message="Mean of empty slice")
warnings.filterwarnings("ignore", message="All-NaN slice encountered")

ROOT = os.path.join(os.path.dirname(__file__), "..")
AQ_BY_NODE_DIR = os.path.join(ROOT, "data", "aq_by_node")
OUT_PATH = os.path.join(ROOT, "data", "pvt_aq_linked.csv")

# Research-staff / test accounts identified in pvt.csv that are not real
# study riders (confirmed 2026-07-29) -- exclude from the cohort.
EXCLUDE_USERNAMES = {"Nussara", "Rach", "Rac", "test", "Preechai", "adisorn", "Patte"}

# Secondary/sensitivity trailing windows, in seconds, preceding each PVT test.
TRAILING_WINDOWS = {
    "1h": 3600,
    "4h": 4 * 3600,
}

# AQ readings are emitted at ~1 reading/second while a sensor is active
# (verified against aq_sensor.csv). Used to convert a summed PM2.5 series
# into a dose-minutes proxy (integrated exposure) for the sameday window.
ASSUMED_SAMPLES_PER_MINUTE = 60


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


EMPTY_STATS = {
    "pm25_mean": np.nan, "pm25_median": np.nan,
    "temp_mean": np.nan, "humidity_mean": np.nan, "n_obs": 0,
}


def _slice_stats(arrs, lo, hi):
    if hi <= lo:
        return dict(EMPTY_STATS)
    pm25 = arrs["pm25"][lo:hi]
    return {
        "pm25_mean": float(np.nanmean(pm25)),
        "pm25_median": float(np.nanmedian(pm25)),
        "temp_mean": float(np.nanmean(arrs["temp"][lo:hi])),
        "humidity_mean": float(np.nanmean(arrs["humidity"][lo:hi])),
        "n_obs": int(hi - lo),
    }


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

    def trailing_window_stats(self, node, ts_end, window_s):
        arrs = self.get(node)
        if arrs is None:
            return dict(EMPTY_STATS)
        ts = arrs["ts"]
        lo = np.searchsorted(ts, ts_end - window_s, side="left")
        hi = np.searchsorted(ts, ts_end, side="right")
        return _slice_stats(arrs, lo, hi)

    def sameday_stats(self, node, ts_end, test_date):
        """Primary exposure window: cumulative from that rider's first AQ
        reading on the test's calendar day (shift start) to the test time.

        Adds shift_duration_min (calendar-elapsed minutes since shift
        start -- the confound-control covariate) and pm25_dose_ugmin (an
        integrated-dose proxy: summed PM2.5 readings / assumed samples
        per minute, i.e. concentration x time, distinct from the plain
        mean concentration in pm25_mean).
        """
        arrs = self.get(node)
        empty = dict(EMPTY_STATS, shift_duration_min=np.nan, pm25_dose_ugmin=np.nan)
        if arrs is None:
            return empty
        day_start = dt.datetime.combine(test_date, dt.time.min).timestamp()
        day_end = dt.datetime.combine(test_date, dt.time.max).timestamp()
        ts = arrs["ts"]
        lo = np.searchsorted(ts, day_start, side="left")
        hi = np.searchsorted(ts, min(day_end, ts_end), side="right")
        if hi <= lo:
            return empty
        stats = _slice_stats(arrs, lo, hi)
        stats["shift_duration_min"] = (ts_end - ts[lo]) / 60.0
        stats["pm25_dose_ugmin"] = float(np.nansum(arrs["pm25"][lo:hi])) / ASSUMED_SAMPLES_PER_MINUTE
        return stats


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
            sameday = cache.sameday_stats(node, rec["timestamp_unix"], row.test_date)
            for k, v in sameday.items():
                rec[f"{k}_sameday"] = v
            for wname, wsec in TRAILING_WINDOWS.items():
                stats = cache.trailing_window_stats(node, rec["timestamp_unix"], wsec)
                for k, v in stats.items():
                    rec[f"{k}_{wname}"] = v
        else:
            for k in ("pm25_mean", "pm25_median", "temp_mean", "humidity_mean", "n_obs",
                      "shift_duration_min", "pm25_dose_ugmin"):
                rec[f"{k}_sameday"] = np.nan
            for wname in TRAILING_WINDOWS:
                for k in ("pm25_mean", "pm25_median", "temp_mean", "humidity_mean", "n_obs"):
                    rec[f"{k}_{wname}"] = np.nan
        records.append(rec)

    out = pd.DataFrame.from_records(records)
    out.to_csv(OUT_PATH, index=False)

    n = len(out)
    covered = out["pm25_mean_sameday"].notna().sum()
    print(f"{n} PVT sessions written to {OUT_PATH}")
    print(f"sameday-window PM2.5 coverage: {covered}/{n} ({100 * covered / n:.1f}%)")


if __name__ == "__main__":
    main()
