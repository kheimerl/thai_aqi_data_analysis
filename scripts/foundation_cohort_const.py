"""Office-worker sensor assignment, confirmed by the user 2026-11-09 (O05 = 6006 accepted as probable)."""
OFFICE_SENSOR = {"Preechai": (6009, "BKK"), "adisorn": (6010, "BKK"), "Nussara": (6033, "BKK"),
                 "Patte": (6007, "CMI"), "Saiyud": (6006, "CMI")}

# Nodes dropped by user decision (Nov 10). Values are kept; class/status mark them.
DROPPED_NODES = {6006: "dropped by user 2026-11-10; do not use"}
DROPPED_PIDS = {"O05": "dropped by user 2026-11-10 (sensor 6006); do not use"}
