"""Build the linked PVT + air-quality-exposure dataset.

For every PVT session by a study rider, attach PM2.5 / temperature /
humidity exposure summaries plus the covariates the analysis plan calls
for (practice/session number, time of day, day of week, rider identity,
shift duration).

Primary exposure window (PLAN.md #3): cumulative same-day, from that
rider's first AQ reading on the test's study day to the test timestamp.
Sensors log 24h on study days, so in practice this is midnight -> test,
not shift start -> test. Secondary/sensitivity windows: 1h and 4h trailing.

Timing: PVT timestamps are naive Asia/Bangkok wall-clock times; AQ
`Timestamp` is true Unix time. All conversions go through LOCAL_TZ
explicitly. Before 2026-09-28 this script read PVT times as UTC and built
day boundaries in the host's zone (America/Los_Angeles), which put every
exposure window 3-7h after the test -- outputs from before that fix are
invalid (see REVIEW_2026-09-28.md).

Requires scripts/split_aq_by_node.sh to have been run first (writes
data/aq_by_node/<node>.csv).

Output: data/pvt_aq_linked.csv
"""
import ast
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

# PVT/peak-flow timestamps are naive wall-clock times in this zone (verified
# 2026-09-28: riders' GPS movement dips around test times only under this
# reading). Thailand has no DST, so this is UTC+7 year-round.
LOCAL_TZ = "Asia/Bangkok"
# Tests before this local hour are late-shift tests just after midnight and
# belong to the previous study day; the next calendar day's AQ export holds
# only a single 00:00 reading.
POST_MIDNIGHT_CUTOFF_H = 4

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
    """Load the rider roster from the summary sheet of the tracking xlsx.

    Returns a DataFrame with one row per (rider, sensor-assignment period):
    PVT user, Node (Grab-<sensor no.>), Start date, End date. A rider can
    appear more than once if they were reassigned to a different sensor
    mid-study (see resolve_node).
    """
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
    """Load pvt.csv, excluding staff/test accounts, and derive per-session
    outcome/covariate columns:

      response_speed_hz -- primary PVT outcome, mean(1000/RT_ms) per
                            session (1/RT handles the right skew in raw
                            reaction times; see PLAN.md #2).
      median_rt_ms       -- secondary outcome.
      error_rate         -- secondary outcome, 1 - percent_success/100.
      session_number     -- that rider's 1-indexed test count over time,
                            controls for practice effects (PLAN.md #4/#5).
      test_date          -- the session's study day: its Bangkok calendar
                            date, except that tests before
                            POST_MIDNIGHT_CUTOFF_H belong to the previous
                            day. Used for AQ joins and day clustering.
      hour_of_day        -- hours since midnight of test_date (Bangkok);
                            exceeds 24 for post-midnight tests.
      day_of_week        -- weekday of test_date.
      timestamp_unix     -- true Unix time of the test (timestamp read as
                            LOCAL_TZ wall-clock, not UTC).

    `timestamp` itself stays the naive Bangkok wall-clock value from pvt.csv.
    Returns one row per PVT session.
    """
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
    local = df["timestamp"].dt.tz_localize(LOCAL_TZ)
    df["timestamp_unix"] = (local - pd.Timestamp("1970-01-01", tz="UTC")).dt.total_seconds()
    df["test_date"] = (local - pd.Timedelta(hours=POST_MIDNIGHT_CUTOFF_H)).dt.date
    study_midnight = pd.to_datetime(df["test_date"]).dt.tz_localize(LOCAL_TZ)
    df["hour_of_day"] = (local - study_midnight).dt.total_seconds() / 3600.0
    df["day_of_week"] = pd.to_datetime(df["test_date"]).dt.day_name()
    return df


EMPTY_STATS = {
    "pm25_mean": np.nan, "pm25_median": np.nan,
    "temp_mean": np.nan, "humidity_mean": np.nan, "n_obs": 0,
}


def _slice_stats(arrs, lo, hi):
    """Summarize a node's PM2.5/temp/humidity arrays over index range
    [lo, hi) into the EMPTY_STATS-shaped dict (mean/median PM2.5, mean
    temp/humidity, and the observation count used as a coverage check).
    Returns all-NaN/zero-count stats if the range is empty."""
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
        """Return {ts, pm25, humidity, temp} numpy arrays (sorted by
        timestamp) for `node`, reading and caching data/aq_by_node/<node>.csv
        on first access. Returns None if that node has no split file."""
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
        """Secondary/sensitivity exposure window (PLAN.md #3): stats over
        the `window_s` seconds immediately preceding `ts_end` (a fixed
        trailing window, e.g. 1h or 4h), regardless of calendar-day
        boundaries."""
        arrs = self.get(node)
        if arrs is None:
            return dict(EMPTY_STATS)
        ts = arrs["ts"]
        lo = np.searchsorted(ts, ts_end - window_s, side="left")
        hi = np.searchsorted(ts, ts_end, side="right")
        return _slice_stats(arrs, lo, hi)

    def sameday_stats(self, node, ts_end, test_date):
        """Primary exposure window: cumulative from that rider's first AQ
        reading at/after Bangkok midnight of the study day (`test_date`)
        to the test time. Not capped at the day's end, so a post-midnight
        test's window spans its whole study day.

        Adds shift_duration_min (elapsed minutes since the window's first
        reading; sensors log 24h, so this is ~minutes since midnight, not
        time on shift) and pm25_dose_ugmin (an integrated-dose proxy:
        summed PM2.5 readings / assumed samples per minute, i.e.
        concentration x time, distinct from the plain mean in pm25_mean).
        """
        arrs = self.get(node)
        empty = dict(EMPTY_STATS, shift_duration_min=np.nan, pm25_dose_ugmin=np.nan)
        if arrs is None:
            return empty
        day_start = pd.Timestamp(test_date).tz_localize(LOCAL_TZ).timestamp()
        ts = arrs["ts"]
        lo = np.searchsorted(ts, day_start, side="left")
        hi = np.searchsorted(ts, ts_end, side="right")
        if hi <= lo:
            return empty
        stats = _slice_stats(arrs, lo, hi)
        stats["shift_duration_min"] = (ts_end - ts[lo]) / 60.0
        stats["pm25_dose_ugmin"] = float(np.nansum(arrs["pm25"][lo:hi])) / ASSUMED_SAMPLES_PER_MINUTE
        return stats


def main():
    """Join every PVT session to its rider's sensor (via resolve_node) and
    attach sameday/1h/4h exposure summaries, writing one row per session
    to data/pvt_aq_linked.csv. Sessions whose rider has no sensor
    assignment covering that date get all-NaN exposure columns (kept in
    the output, not dropped, so per-analysis scripts can decide how to
    handle missingness)."""
    roster = load_roster()
    pvt = load_pvt()
    cache = NodeCache()

    records = []
    for row in pvt.itertuples(index=False):
        node = resolve_node(row.username, row.test_date, roster)
        rec = row._asdict()
        rec["node"] = node

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
