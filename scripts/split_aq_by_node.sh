#!/usr/bin/env bash
# One-time preprocessing: split the 5GB aq_sensor.csv into one CSV per Node.
#
# aq_sensor.csv is too large to scan repeatedly per rider/session. This
# splits it once into data/aq_by_node/<node>.csv so later steps only read
# the ~27 per-node files relevant to a given rider instead of the full file.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$ROOT/aq_sensor.csv"
OUT_DIR="$ROOT/data/aq_by_node"

mkdir -p "$OUT_DIR"
HEADER="$(head -1 "$SRC")"

awk -F, -v out_dir="$OUT_DIR" -v header="$HEADER" '
NR == 1 { next }
{
  node = $2
  file = out_dir "/" node ".csv"
  if (!(node in seen)) {
    seen[node] = 1
    print header > file
  }
  print > file
}
END {
  for (n in seen) close(out_dir "/" n ".csv")
}
' "$SRC"

echo "done: split into $(ls "$OUT_DIR" | wc -l) per-node files -> $OUT_DIR"
