# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

This is a **data directory, not a codebase** — there is no source code, build system, or test suite here yet. It holds raw and lightly-processed data from a field study of Bangkok motorcycle-taxi ("Grab") drivers wearing air-quality sensors, part of the air-quality/motorcycle-taxi-driver research thread (see Nussara "Firn" Tieanklin's work in the broader research context). Any future work in this directory will likely involve writing analysis scripts/notebooks against these files — there is no existing convention to follow yet, so establish one (e.g. a `scripts/` or `notebooks/` dir) rather than dropping loose files in the root.

## Environment

`python3-pandas` (2.1.4, apt-installed system-wide) is available. This system uses Debian's externally-managed-environment guard, so additional packages should be installed via `apt install python3-<pkg>` where a Debian package exists, rather than `pip install` (which fails with `externally-managed-environment` unless a venv is used).

## Data files

### `2025-2026 Rider-Sensor-Summary.xlsx` — the roster / join key
The rosetta stone linking every other file together. Two sheets:
- **`summary`**: one row per rider — `Region`, `Rider name` (Thai), `LINE name` (their handle in the PVT/peak-flow apps — matches `username` in `pvt.csv`/`peakflow.csv`), `PVT user`, `Sensor no.` (matches the numeric suffix of `Node` in the AQ sensor data, e.g. sensor `6011` → `Node` = `Grab-6011`), `Start date`/`End date` of deployment (stored as **Excel serial dates**, not ISO strings).
- **`sensor data availability`**: one row per rider, one column per week (headers are Excel serial dates), cell values `y`/`n` indicating whether that rider's sensor reported data that week.

To join AQ readings, PVT results, and peak-flow results for a given rider, go through this roster — `username` (PVT/peakflow) and `Node`/sensor number (AQ) are **not** directly comparable strings.

### `aq_sensor.csv` (~5.1 GB) / `aq_sensor_csv.zip` (~800 MB, same content zipped)
Raw per-second air-quality telemetry from the `Grab-XXXX` sensor nodes. Columns:
`Timestamp` (Unix seconds), `Node` (`Grab-XXXX`), `Datetime(UTC+7)`, `GPS_Lat`, `GPS_Lon`, `GPS_Alt` (frequently blank), `Relative humidity`, `PM2.5`, `Temperature`.
This file is too large to load into memory naively even with pandas installed (see below) — read it in chunks (`pd.read_csv(..., chunksize=...)`) or filter by `Node`/`Timestamp` with `awk`/`grep` before pulling data into a DataFrame. Consider `duckdb`/`polars` for out-of-core queries if the analysis needs more than a chunked scan. The `.zip` is a redundant compressed copy of the same CSV — don't process both.

### `aq_sensor_count.txt`
Precomputed row count per `Node` for `aq_sensor.csv` (`Count, Node`), useful as a cheap sanity check / index without scanning the full CSV.

### `pvt.csv` (~2,100 rows)
Results from a **Psychomotor Vigilance Task** (a standard sustained-attention/fatigue test), one row per test administration: `timestamp`, `username` (LINE name — join via the roster), `valid_taps`, `total_taps`, `percent_success`, `average_response_time`, plus three columns holding **stringified Python-style lists** (`response_times`, `stimulation_times`, `all_tap_times`) — these need `ast.literal_eval` or similar to parse, not plain CSV parsing of the field.

### `peakflow.csv` (~1,300 rows)
Respiratory peak-expiratory-flow readings: `timestamp`, `username` (join via roster), `peak_flow`.

## Gotchas
- `username` values are informal handles (emoji, mixed Thai/English) — match against the roster's `LINE name` column, not `Rider name`.
- Dates in the xlsx are Excel serial numbers; convert with an epoch of 1899-12-30.
- GPS fields in `aq_sensor.csv` are mostly empty — don't assume location data is reliably present.
- `aq_sensor.csv` and `aq_sensor_csv.zip` are duplicates of the same data; only one needs to be read.
