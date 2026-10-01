# 2026-09 re-analysis

Scripts behind `REVIEW_2026-09-28.md` and `REANALYSIS_SPEC_2026-09-28.md` (repo root).
The headline: the original PM2.5 → PVT result came from a timezone bug in
`scripts/build_pvt_aq_dataset.py` (now fixed). On corrected exposure there is no
detectable acute effect, and the interval bounds it at a few ms per +50 µg/m³.

Run everything with `run_all.sh` (~15 min), after the fixed pipeline has written
`data/pvt_aq_linked.csv`. Every script runs inside `data/review_2026_09/` (via
`_paths.py`) and writes its log to `data/review_2026_09/logs/`. The pre-fix outputs
are kept in `data/superseded_buggy_tz/`. Both directories are untracked, like the
rest of `data/`.

| Script | What it shows | Status |
| --- | --- | --- |
| `fetch_ambient.py` | ERA5 + METAR (VTBD, VTCC) ambient temperature, city centres only | data |
| `minute_table.py`, `gps_days.py` | Per-minute AQ table with GPS displacement; per-day GPS coverage | data |
| `verify_tz.py`, `verify_tz2.py` | Stop-to-test movement dip appears only if PVT times are Bangkok-local | check |
| `build_corrected.py` | Corrected linked data + placebo/onset windows; reproduces the bug exactly | data, check |
| `run_reanalysis.py` | R0–R2, S1–S2, N1–N2 from the spec | declared |
| `multiverse.py` | Specification curve, pre-test vs post-test (placebo) windows | post-hoc |
| `diagnostics.py`, `region_split.py`, `region_sign.py` | Effect size; session_number decline; Bangkok/Chiang Mai sign flip | post-hoc |
| `offshift.py`, `home_road.py`, `temp_vs_ambient.py` | Off-shift data volume; home vs road PM2.5; sensor vs ambient temperature | post-hoc |
| `within_day.py`, `remove_drift.py`, `adjacent_days.py` | Within-day, drift-removed and adjacent-day designs | post-hoc |
| `bounds_and_variance.py` | ms bounds, between-day variance, dose split, cross-rider dependence | post-hoc |

Declared analyses carry the conclusion; post-hoc ones are hypothesis-generating.
