# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Data and analysis scripts from a field study of motorcycle-taxi ("Grab") riders in Bangkok and Chiang Mai wearing air-quality sensors, part of the air-quality/motorcycle-taxi-driver research thread (see Nussara "Firn" Tieanklin's work in the broader research context). Git tracks only the root `*.md` files and `scripts/`; raw data and everything under `data/` stay local.

- `scripts/split_aq_by_node.sh` → `scripts/build_pvt_aq_dataset.py` → `data/pvt_aq_linked.csv`: the linking pipeline.
- `scripts/fit_*.py`: the July–August 2026 analyses (PLAN.md §4–§6). Their PM2.5 results were produced before the 2026-09-28 timezone fix and are superseded; outputs from that run are in `data/superseded_buggy_tz/`.
- `scripts/review_2026_09/`: the corrected re-analysis (see its README, `REVIEW_2026-09-28.md` and `REANALYSIS_SPEC_2026-09-28.md`).

New analysis code goes under `scripts/`, not loose in the root.

## Environment

`python3-pandas` (2.1.4, apt-installed system-wide) is available. This system uses Debian's externally-managed-environment guard, so additional packages should be installed via `apt install python3-<pkg>` where a Debian package exists, rather than `pip install` (which fails with `externally-managed-environment` unless a venv is used).

## Data files

### `2025-2026 Rider-Sensor-Summary.xlsx` — the roster / join key
The rosetta stone linking every other file together. Two sheets:
- **`summary`**: one row per rider-sensor assignment — `Region` (Bangkok or Chiang mai), `Rider name` (Thai), `LINE name`, `PVT user` (matches `username` in `pvt.csv`/`peakflow.csv`), `Sensor no.` (matches the numeric suffix of `Node` in the AQ data, e.g. sensor `6011` → `Node` = `Grab-6011`), `Start date`/`End date` of deployment. `pd.read_excel` parses these as datetimes.
- **`sensor data availability`**: one row per rider, one column per week (headers are Excel serial dates), cell values `y`/`n` indicating whether that rider's sensor reported data that week.

To join AQ readings, PVT results, and peak-flow results for a given rider, go through this roster — `username` (PVT/peakflow) and `Node`/sensor number (AQ) are **not** directly comparable strings. Joins must use `(PVT user, sensor, date range)`: sensors 6021, 6022 and 6029/6032 changed hands or were replaced mid-study.

### `aq_sensor.csv` (~5.1 GB) / `aq_sensor_csv.zip` (~800 MB, same content zipped)
Per-second air-quality telemetry from the `Grab-XXXX` sensor nodes. Columns:
`Timestamp` (true Unix seconds), `Node` (`Grab-XXXX`), `Datetime(UTC+7)` (Bangkok wall-clock), `GPS_Lat`, `GPS_Lon`, `GPS_Alt` (frequently blank), `Relative humidity`, `PM2.5`, `Temperature`.
- The export covers only the ~35 study days (all weekdays), each a full 00:00–24:00 at ~1 Hz, plus a single 00:00 reading the next day. Sensors log overnight at home, so "first reading of the day" is midnight, not shift start.
- Nodes 6006–6010 and 6033 are not in the roster.
- Too large to load naively — read it in chunks or filter by `Node` first; `split_aq_by_node.sh` writes `data/aq_by_node/<node>.csv`. The `.zip` is a redundant copy — don't process both.

### `aq_sensor_count.txt`
Precomputed row count per `Node` for `aq_sensor.csv` (`Count, Node`), useful as a cheap sanity check / index without scanning the full CSV.

### `pvt.csv` (~2,100 rows)
Results from a **Psychomotor Vigilance Task** (a 5-minute, ~100-trial phone version), one row per test administration: `timestamp`, `username` (join via the roster's `PVT user`), `valid_taps`, `total_taps`, `percent_success`, `average_response_time`, plus three columns holding **stringified Python-style lists** (`response_times`, `stimulation_times`, `all_tap_times`) — these need `ast.literal_eval` or similar to parse, not plain CSV parsing of the field.

### `peakflow.csv` (~1,300 rows)
Respiratory peak-expiratory-flow readings: `timestamp`, `username` (join via the roster's `PVT user`), `peak_flow`. Readings cluster at 08–09h and 21–22h (morning/evening). Not yet analyzed.

## Gotchas
- **Timezones.** `pvt.csv`/`peakflow.csv` timestamps are naive **Asia/Bangkok** wall-clock times, verified against GPS movement (riders stop to take the test). Localize them before comparing with AQ `Timestamp`. Never call `.timestamp()` on the naive value (pandas assumes UTC) or build day boundaries with naive `datetime` (uses the host's Pacific zone) — that was the bug that invalidated the first analysis. Tests before 04:00 belong to the previous study day.
- `username` values are informal handles (emoji, mixed Thai/English); they match the roster's `PVT user`, not `LINE name` or `Rider name`.
- GPS fixes are present nearly all day on the rider nodes (6011–6032), even at home. Fix presence does not mean riding; use displacement (100–2000 m per minute) to find movement and shifts.
- Sensor `Temperature`/`Relative humidity` are device readings, not ambient: about +3 °C vs airport METAR on the road and +5–8 °C at home. Use METAR (VTBD Bangkok, VTCC Chiang Mai) or ERA5 for ambient temperature (`scripts/review_2026_09/fetch_ambient.py`).
- Bangkok and Chiang Mai have opposite haze seasons over the study (Bangkok peaks in January, Chiang Mai's burning season in March–April). Shared day-level variation dominates exposure, so compare riders within the same city and day.
- `aq_sensor.csv` and `aq_sensor_csv.zip` are duplicates of the same data; only one needs to be read.
