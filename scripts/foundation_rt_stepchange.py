"""v4.1 item 6 (2026-11-10): per-rider step-change check on session median RT (exposure-free).
Series: in-enrolment rider sessions without P1/P2 flags, in time order; y = session median RT (RT >= 100 ms).
For each split i with >= K sessions on each side: shift = median(next K) - median(previous K).
Reported: largest |shift| (ms, % of the preceding median), date of the first session after it, a z-score
(shift / (1.2533 * robust SD * sqrt(2/K)), robust SD from first differences / sqrt(2) so trends don't inflate it),
persistence (median of ALL sessions after - ALL before), and the same scan after removing a per-rider
log(session number) practice trend. DESCRIPTIVE: 'possible_step' = |z| >= 4 and |shift| >= 10% on BOTH
the raw and the detrended series. To be checked against the team's knowledge of phone changes.
No quantization fingerprint of devices was found in the RTs (ms values look continuous)."""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from foundation_common import *
K = 10
q = pd.read_csv(os.path.join(FOUND, "qc_session_flags.csv"))
q = q[(q.role == "rider") & ~q.P11_outside_enrolment & ~q.P1_incomplete & ~q.P2_low_accuracy].copy()
q["t"] = pd.to_datetime(q.timestamp); q = q.sort_values(["pid", "t"])


def scan(y, dates):
    best = (0, None, np.nan, np.nan)
    for i in range(K, len(y) - K + 1):
        a, b = np.median(y[i - K:i]), np.median(y[i:i + K])
        if abs(b - a) > abs(best[0]): best = (b - a, i, a, b)
    return best


rows = []
for pid, g in q.groupby("pid"):
    y = g.med_rt.values.astype(float); dates = g.t.dt.strftime("%Y-%m-%d").values
    n = len(y)
    if n < 2 * K: rows.append(dict(pid=pid, n_sessions=n, note="too few sessions")); continue
    rsd_d = 1.4826 * np.median(np.abs(np.diff(y) - np.median(np.diff(y)))) / np.sqrt(2)
    se = 1.2533 * rsd_d * np.sqrt(2 / K)
    s, i, a, b = scan(y, dates)
    X = np.column_stack([np.ones(n), np.log(np.arange(1, n + 1))]); beta = np.linalg.lstsq(X, y, rcond=None)[0]
    yd = y - X @ beta + np.median(y)
    sd_, idd, ad, bd = scan(yd, dates)
    rows.append(dict(pid=pid, n_sessions=n, median_rt_all=round(np.median(y), 1),
                     largest_shift_ms=round(s, 1), largest_shift_pct=round(100 * s / a, 1), shift_date=dates[i],
                     shift_session_index=int(i + 1), median_before_K=round(a, 1), median_after_K=round(b, 1),
                     z=round(s / se, 2) if se > 0 else np.nan,
                     persistent_shift_ms=round(np.median(y[i:]) - np.median(y[:i]), 1),
                     detrended_shift_ms=round(sd_, 1), detrended_shift_pct=round(100 * sd_ / ad, 1),
                     detrended_shift_date=dates[idd], detrended_z=round(sd_ / se, 2) if se > 0 else np.nan))
R = pd.DataFrame(rows)
R["possible_step"] = (R.z.abs() >= 4) & (R.largest_shift_pct.abs() >= 10) & (R.detrended_z.abs() >= 4) & (R.detrended_shift_pct.abs() >= 10)
R.to_csv(os.path.join(FOUND, "rt_stepchange.csv"), index=False)
print(R.drop(columns=["shift_session_index"]).to_string())
