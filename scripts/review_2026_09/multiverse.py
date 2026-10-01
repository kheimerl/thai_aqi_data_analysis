"""Specification curve over defensible analytic choices, run identically on
pre-test exposure windows and on post-test (placebo) windows. If the pre-test
curve is no more negative / no more often significant than the placebo curve,
the PM2.5 association is not specific to air the rider had actually breathed."""
import _paths  # side effect: chdir into data/review_2026_09/
import itertools
import warnings

import numpy as np
import pandas as pd

from run_reanalysis import per_rider, pool

warnings.filterwarnings("ignore")

df = pd.read_csv("linked_corrected.csv")
df["study_day_num"] = (pd.to_datetime(df.study_day) - pd.Timestamp("2025-12-12")).dt.days

WINDOWS = {  # name: (pm25 col, temp col, rh col, time-on-shift col, kind)
    "pre 1h": ("pm25_mean_1h", "temp_mean_1h", "humidity_mean_1h", "shift_duration_min_sameday", "pre"),
    "pre 4h": ("pm25_mean_4h", "temp_mean_4h", "humidity_mean_4h", "shift_duration_min_sameday", "pre"),
    "pre midnight->test": ("pm25_mean_sameday", "temp_mean_sameday", "humidity_mean_sameday", "shift_duration_min_sameday", "pre"),
    "pre onset->test": ("pm25_mean_onset", "temp_mean_onset", "humidity_mean_onset", "onset_to_test_min", "pre"),
    "post 1h": ("pm25_mean_post1h", "temp_mean_post1h", "humidity_mean_post1h", "shift_duration_min_sameday", "placebo"),
    "post 4h": ("pm25_mean_post4h", "temp_mean_post4h", "humidity_mean_post4h", "shift_duration_min_sameday", "placebo"),
    "post test->midnight": ("pm25_mean_postday", "temp_mean_postday", "humidity_mean_postday", "shift_duration_min_sameday", "placebo"),
}
OUTCOMES = {"raw 1/RT": "response_speed_hz", "1/RT, false starts removed": "response_speed_clean_hz"}
SES = ["CR1", "CR2"]
POOLS = ["DL+Wald", "REML+HKSJ"]
TIME = {"session_number": ["session_number"], "session_number+study_day": ["session_number", "study_day_num"]}

rows = []
for (wn, (pm, tc, rc, sc, kind)), (on, oc), se, pl, (tn, tcols) in itertools.product(
        WINDOWS.items(), OUTCOMES.items(), SES, POOLS, TIME.items()):
    t = per_rider(df, oc, [pm], [tc, rc, sc] + tcols + ["hour_of_day"], se_type=se)
    r = pool(t[f"b_{pm}"], t[f"se_{pm}"], pl)
    rows.append(dict(window=wn, kind=kind, outcome=on, se=se, pooling=pl, time=tn,
                     k=r["k"], beta=r["mu"] * 1e4, lo=r["lo"] * 1e4, hi=r["hi"] * 1e4, p=r["p"], i2=r["i2"]))
mv = pd.DataFrame(rows)
mv.to_csv("multiverse.csv", index=False)

pd.set_option("display.width", 200)
print("Specification curve: per-window summary over 16 specs each (beta x1e4)")
s = mv.groupby(["kind", "window"]).agg(median_beta=("beta", "median"), min_beta=("beta", "min"), max_beta=("beta", "max"),
                                       n_sig=("p", lambda p: int((p < 0.05).sum())), n=("p", "size"),
                                       median_p=("p", "median")).round(3)
print(s.to_string())
for kind in ("pre", "placebo"):
    k = mv[mv.kind == kind]
    print(f"{kind:8s}: {int((k.p < .05).sum())}/{len(k)} specs p<.05; median beta {k.beta.median():+.2f}; "
          f"{int((k.beta < 0).sum())}/{len(k)} negative")
print("\nPrimary-window (pre midnight->test) specs:")
print(mv[mv.window == "pre midnight->test"][["outcome", "se", "pooling", "time", "beta", "lo", "hi", "p"]].round(3).to_string(index=False))
