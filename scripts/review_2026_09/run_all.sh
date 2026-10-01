#!/usr/bin/env bash
# Re-run the 2026-09 re-analysis end to end (~15 min). Requires the fixed
# scripts/build_pvt_aq_dataset.py to have been run (data/pvt_aq_linked.csv),
# and the pre-fix outputs in data/superseded_buggy_tz/ for the harness and
# original-vs-corrected comparisons. Each script's output goes to
# data/review_2026_09/logs/<script>.txt.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOGS="$HERE/../../data/review_2026_09/logs"
mkdir -p "$LOGS"
for s in fetch_ambient minute_table gps_days verify_tz verify_tz2 build_corrected \
         run_reanalysis multiverse diagnostics region_split offshift home_road \
         temp_vs_ambient region_sign within_day remove_drift adjacent_days bounds_and_variance; do
  echo "== $s"
  python3 "$HERE/$s.py" | tee "$LOGS/$s.txt"
done
