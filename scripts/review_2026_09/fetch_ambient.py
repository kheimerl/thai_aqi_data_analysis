"""Fetch ambient temperature for the study period into data/review_2026_09/.

- ERA5 hourly 2 m temperature/RH at each city centre (Open-Meteo archive API)
  -> era5_bangkok.json, era5_chiangmai.json
- Airport METAR temperature/RH (Iowa Environmental Mesonet ASOS archive):
  VTBD Bangkok Don Mueang, VTCC Chiang Mai -> metar_VTBD.csv, metar_VTCC.csv

Only city-centre coordinates and station codes are sent, never rider GPS.
Existing files are kept unless --force is given. IEM rate-limits rapid
requests, hence the pause between stations.
"""
import _paths  # side effect: chdir into data/review_2026_09/
import os
import sys
import time
import urllib.request

START, END = "2025-12-10", "2026-05-01"
ERA5 = {"bangkok": (13.7563, 100.5018), "chiangmai": (18.7883, 98.9853)}
METAR = ["VTBD", "VTCC"]


def fetch(url, path, force):
    """Download `url` to `path` unless it exists (or `force`); return True if fetched."""
    if os.path.exists(path) and not force:
        print(f"keep  {path}")
        return False
    with urllib.request.urlopen(url, timeout=120) as r:
        body = r.read()
    if b"Too many requests" in body[:200]:
        raise RuntimeError(f"rate-limited fetching {path}; wait a minute and rerun")
    with open(path, "wb") as f:
        f.write(body)
    print(f"wrote {path} ({len(body):,} bytes)")
    return True


def main():
    force = "--force" in sys.argv
    for name, (lat, lon) in ERA5.items():
        fetch(f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}"
              f"&start_date={START}&end_date={END}&hourly=temperature_2m,relative_humidity_2m"
              f"&timezone=Asia%2FBangkok", f"era5_{name}.json", force)
    y1, m1, d1 = START.split("-")
    y2, m2, d2 = END.split("-")
    for i, st in enumerate(METAR):
        if fetch(f"https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?station={st}&data=tmpc&data=relh"
                 f"&year1={y1}&month1={int(m1)}&day1={int(d1)}&year2={y2}&month2={int(m2)}&day2={int(d2) + 1}"
                 f"&tz=Asia%2FBangkok&format=onlycomma&latlon=no&missing=M&trace=T&direct=no"
                 f"&report_type=3&report_type=4", f"metar_{st}.csv", force) and i < len(METAR) - 1:
            time.sleep(30)


if __name__ == "__main__":
    main()
