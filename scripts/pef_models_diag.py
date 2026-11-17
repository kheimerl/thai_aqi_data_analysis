"""Post-hoc diagnostics for Addendum D (2026-11-17; added after seeing pef_models.py output, labelled as such).
Why do rider-FE models differ from the two-stage estimates? (1) FE + city x month dummies (opposite city seasons);
(2) FE with rider-specific slopes for every covariate (as the two-stage model has), common PM slope.
python3 -I scripts/pef_models_diag.py --pm-col pm  -> peakflow_paper/run_models/diag.json
"""
import argparse, json, os, sys
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pef_analysis import load, select, ROOT  # noqa: E402
from pef_models import crfit, prep  # noqa: E402


def fe_design(df, y, ctrls, city_month=False, rider_slopes=False):
    cols = ["xsd"]
    X = pd.DataFrame({"xsd": df["xsd"]})
    for c in ctrls:
        if rider_slopes:
            for p in df.pid.unique():
                X[f"{c}_{p}"] = df[c].where(df.pid == p, 0.0)
        else:
            X[c] = df[c]
    if city_month:
        dm = pd.get_dummies(df.city_c + "_" + df.month, drop_first=True, dtype=float)
        X = pd.concat([X, dm], axis=1)
    X = X - X.groupby(df.pid).transform("mean"); yy = df[y] - df.groupby("pid")[y].transform("mean")
    X = X.loc[:, X.abs().sum() > 1e-9]
    names = ["x"] + [None] * (X.shape[1] - 1)
    r, info = crfit(yy.to_numpy(float), X.to_numpy(float), df.pid.to_numpy(), names, absorbed=df.pid.nunique())
    return dict(**r["x"], **info, params=int(X.shape[1]))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--pm-col", default="pm"); a = ap.parse_args()
    d, _ = load(a.pm_col); s, _ = select(d, "highest")
    ev = s[s.period == "evening"].copy(); mo = s[s.period == "morning"].copy()
    C = ["clock_u", "hours_on_shift_before_test"]
    df, _ = prep(ev, "pct", "PM_sd", C)
    mo["PM_night"] = mo[f"{a.pm_col}_night00"].where(mo.pm_night00_nmin >= 180); mo = mo[mo.pm_after24_nmin >= 180]
    CN = ["days_since_enrolment", "clock_u"]
    dn, _ = prep(mo, "pct", "PM_night", CN)
    R = dict(L1_fe_citymonth=fe_design(df, "pct", C, city_month=True),
             L1_fe_riderslopes=fe_design(df, "pct", C, rider_slopes=True),
             L1_fe_both=fe_design(df, "pct", C, city_month=True, rider_slopes=True),
             night_fe_citymonth=fe_design(dn, "pct", CN, city_month=True),
             night_fe_riderslopes=fe_design(dn, "pct", CN, rider_slopes=True),
             night_fe_both=fe_design(dn, "pct", CN, city_month=True, rider_slopes=True))
    json.dump(R, open(os.path.join(ROOT, "peakflow_paper/run_models/diag.json"), "w"), indent=1)
    for k, v in R.items():
        print(f"{k:22s} {v['b']:+.2f} [{v['lo']:+.2f}, {v['hi']:+.2f}] p {v['p']:.3f} n {v['n']} params {v['params']}")


if __name__ == "__main__":
    main()
