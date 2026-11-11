"""Person IDs: R01-R22 riders (BKK by sensor, then CMI by sensor/start), O01-O05 office.
Writes foundation/_private_key.csv (handle <-> ID; untracked, never quote)."""
import os, sys
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from foundation_common import *
from foundation_task1_reproduce import roster

OFFICE = ["Preechai", "adisorn", "Nussara", "Patte", "Saiyud"]
TEST = ["Rac", "Rach", "test"]


def person_table():
    m = read_mapping(); ro = roster()
    first = ro.groupby("user").agg(start=("start", "min"), end=("end", "max"), sensor0=("sensor", "first"))
    m = m.merge(first, left_on="user", right_index=True, how="left")
    m["city_order"] = m.city.map({"BKK": 0, "CMI": 1})
    m = m.sort_values(["city_order", "sensor", "start"]).reset_index(drop=True)
    m["pid"] = [f"R{i+1:02d}" for i in range(len(m))]
    r = m[["pid", "user", "city", "start", "end"]].assign(role="rider")
    o = pd.DataFrame({"pid": [f"O{i+1:02d}" for i in range(len(OFFICE))], "user": OFFICE, "role": "office"})
    return pd.concat([r, o], ignore_index=True)


def assignments():
    """(pid, sensor, start, end) from roster (riders)."""
    pt = person_table(); ro = roster()
    a = ro.merge(pt[["pid", "user"]], on="user")
    return a[["pid", "sensor", "start", "end"]]


if __name__ == "__main__":
    pt = person_table()
    pt[["pid", "user", "role"]].to_csv(os.path.join(FOUND, "_private_key.csv"), index=False)
    print(pt[["pid", "city", "role", "start", "end"]].to_string())
    print(assignments().to_string())
