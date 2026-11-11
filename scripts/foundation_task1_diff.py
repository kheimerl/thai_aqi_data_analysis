"""S1 Task 1 (after own build): attribute the beta gap vs analysis_v3 step by step."""
import os, sys, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from foundation_common import *
W = os.path.join(FOUND, "_work")
mine = pd.read_csv(os.path.join(W, "task1_sessions_BKK.csv"))
v3 = pd.read_csv(os.path.join(ROOT, "analysis_v3", "pvt_linked_bangkok.csv")); v3["user"] = v3.username.str.strip()
j = mine.merge(v3[["user", "timestamp", "y1", "pm", "temp", "rh", "shift", "session_number", "hour_of_day"]],
               on=["user", "timestamp"], how="left")

def run(d, ycol, covs, min_n=10, label=""):
    rows = []
    for u, g in d.groupby("user"):
        g = g.dropna(subset=[ycol, "pm_sameday"] + covs)
        if len(g) < min_n or g.pm_sameday.std() == 0: continue
        X = np.column_stack([np.ones(len(g)), g.pm_sameday / 10] + [g[c] for c in covs])
        b, se = ols(g[ycol], X); rows.append((b[1], se[1], len(g)))
    b, se, n = map(np.array, zip(*rows)); P = pool_dl_hksj(b, se)
    return dict(step=label, k=P["k"], n=int(n.sum()), beta=round(P["beta"], 5), ci_hksj=[round(x, 4) for x in P["ci_hksj"]],
                p_hksj=round(P["p_hksj"], 3), p_dl=round(P["p_dl"], 3))

out = []
A = j[j.pm_sameday.notna()]
out.append(run(A, "speed", ["day_of_study"], label="A mine: RT>=100, any PM, PM+day"))
B = A[A.pm_n_sec >= 60]
out.append(run(B, "speed", ["day_of_study"], label="B + >=60 PM samples (v3 rule)"))
out.append(run(B, "y1", ["day_of_study"], label="C + v3 speed (RT>0)"))
covs = ["temp", "rh", "shift", "session_number", "hour_of_day"]
out.append(run(B, "y1", covs, label="D + v3 covariates (temp,RH,shift,session#,hour), no day"))
out.append(run(B, "y1", covs, min_n=15, label="E + min 15 sessions/rider (classical SE; v3 uses cluster/HC1 SE + IntHout floor)"))
out.append(run(B, "speed", covs + ["day_of_study"], label="F mine speed, v3 covs + day"))
for o in out: print(json.dumps(o, ensure_ascii=False))
pd.DataFrame(out).to_csv(os.path.join(W, "task1_attribution.csv"), index=False)
