"""METAR-only download for SPEC_ADDENDUM §C (heat as a named rival), approved by the user 2026-11-12 (C.7-1 option b).

Fetches hourly airport air temperature (and RH) for VTBD (Bangkok Don Mueang) and VTCC (Chiang Mai) from the
Iowa Environmental Mesonet ASOS archive, Dec 10 2025 - May 1 2026, times in Asia/Bangkok local.
Only station codes and dates are sent; no rider data. No ERA5 / Open-Meteo request.
Writes data/review_2026_09/metar_<station>.csv; existing files are kept unless --force.
Same request as scripts/review_2026_09/fetch_ambient.py (METAR part).
Usage: python3 -I scripts/pvt_fetch_metar.py [--force]
"""
import os, sys, time, urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data", "review_2026_09")
START, END = (2025, 12, 10), (2026, 5, 2)

def main():
    force = "--force" in sys.argv
    os.makedirs(OUT, exist_ok=True)
    for i, st in enumerate(["VTBD", "VTCC"]):
        path = os.path.join(OUT, f"metar_{st}.csv")
        if os.path.exists(path) and not force:
            print("keep", path); continue
        url = ("https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?"
               f"station={st}&data=tmpc&data=relh&year1={START[0]}&month1={START[1]}&day1={START[2]}"
               f"&year2={END[0]}&month2={END[1]}&day2={END[2]}&tz=Asia%2FBangkok&format=onlycomma"
               "&latlon=no&missing=M&trace=T&direct=no&report_type=3&report_type=4")
        with urllib.request.urlopen(url, timeout=120) as r:
            body = r.read()
        if b"Too many requests" in body[:200]:
            raise RuntimeError("IEM rate limit; wait a minute and rerun")
        open(path, "wb").write(body)
        print(f"wrote {path} ({len(body):,} bytes)")
        if i == 0: time.sleep(30)

if __name__ == "__main__":
    main()
