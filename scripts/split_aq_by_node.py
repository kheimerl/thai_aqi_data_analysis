"""One-time preprocessing: split the 5GB aq_sensor.csv into one CSV per Node.

aq_sensor.csv is too large to scan repeatedly per rider/session. This splits it
once into data/aq_by_node/<node>.csv so later steps only read the ~27
per-node files relevant to a given rider instead of the full file.
"""
import csv
import os

SRC = os.path.join(os.path.dirname(__file__), "..", "aq_sensor.csv")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "aq_by_node")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    writers = {}
    files = {}
    header = None
    n = 0
    with open(SRC, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            node = row[1]
            if node not in writers:
                fh = open(os.path.join(OUT_DIR, f"{node}.csv"), "w", newline="")
                w = csv.writer(fh)
                w.writerow(header)
                files[node] = fh
                writers[node] = w
            writers[node].writerow(row)
            n += 1
            if n % 5_000_000 == 0:
                print(f"  {n:,} rows processed")
    for fh in files.values():
        fh.close()
    print(f"done: {n:,} rows split across {len(writers)} nodes -> {OUT_DIR}")


if __name__ == "__main__":
    main()
