"""cases/ R21 haze-season comparison (SPEC addendum 2026-11-16). Exploratory, descriptive, no tests.
Run after case_build.py: python3 -I scripts/case_r21_haze.py [--pm-col pm]
"""
import argparse, os
import numpy as np, pandas as pd
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
F = lambda *p: os.path.join(ROOT, *p)
ap = argparse.ArgumentParser(); ap.add_argument('--pm-col', default='pm'); A = ap.parse_args()
tb = lambda s: s.astype(str).str.lower().isin(['true', '1', '1.0'])
season = lambda m: np.where(pd.Series(m).isin([12, 1, 2]), 'DecFeb', 'MarApr')

M = pd.read_pickle(F('cases', '_work', 'minutes.pkl')); M = M[M.city == 'CMI']
M['date'] = pd.to_datetime(M.minute_unix + 7 * 3600, unit='s').dt.normalize()
M['season'] = season(M.workday.dt.month)
b = M[M.both]
rows = []
for (p, s), g in b.groupby(['pid', 'season']):
    nt = g[g.night_home].groupby('date')[['pm', 'ref']].sum().assign(n=g[g.night_home].groupby('date').size()); nt = nt[nt.n >= 120]
    on = g[g.onshift].groupby('workday')[['pm', 'ref']].sum().assign(n=g[g.onshift].groupby('workday').size()); on = on[on.n >= 60]
    rows.append({'pid': p, 'season': s, 'night_ratio': nt.pm.sum() / nt.ref.sum() if len(nt) else np.nan, 'n_nights': len(nt),
                 'onshift_ratio': on.pm.sum() / on.ref.sum() if len(on) else np.nan, 'n_shift_days': len(on),
                 'own_night_pm': nt.pm.sum() / nt.n.sum() if len(nt) else np.nan, 'own_on_pm': on.pm.sum() / on.n.sum() if len(on) else np.nan})
X = pd.DataFrame(rows).pivot(index='pid', columns='season')
X.columns = [f'{a}_{c}' for a, c in X.columns]
X['own_night_MarApr_vs_DecFeb'] = X.own_night_pm_MarApr / X.own_night_pm_DecFeb
X['own_onshift_MarApr_vs_DecFeb'] = X.own_on_pm_MarApr / X.own_on_pm_DecFeb
X = X.drop(columns=[c for c in X.columns if c.startswith('own_') and ('_pm_' in c)])

PF = pd.read_csv(F('foundation', 'linked_v4_pef.csv'), low_memory=False)
PF = PF[(PF.role == 'rider') & (PF.city == 'CMI') & ~tb(PF.exact_duplicate) & ~tb(PF.any_exclusion) & PF.period.isin(['morning', 'evening'])].copy()
PF['workday'] = pd.to_datetime(PF.workday); PF['season'] = season(PF.workday.dt.month)
pf = PF.sort_values('peak_flow').groupby(['pid', 'workday', 'period']).last().reset_index()
w = pf.pivot_table(index=['pid', 'workday', 'season'], columns='period', values='pct_pb').reset_index(); w['drop'] = w.evening - w.morning
pe = w.groupby(['pid', 'season']).agg(morning=('morning', 'mean'), evening=('evening', 'mean'), drop=('drop', 'mean'), n_pef_days=('morning', 'size')).unstack('season')
pe.columns = [f'{a}_{c}' for a, c in pe.columns]
V = pd.read_csv(F('foundation', 'linked_v4_pvt.csv'), low_memory=False)
V = V[(V.role == 'rider') & (V.city == 'CMI') & ~tb(V.any_exclusion) & V.y1_speed_rt100.notna()].copy()
V['season'] = season(pd.to_datetime(V.workday).dt.month)
pv = V.groupby(['pid', 'season']).y1_speed_rt100.agg(['mean', 'size']).unstack('season'); pv.columns = [f'y1_{a}_{c}' for a, c in pv.columns]
R = X.join(pe).join(pv)
okp = (R.n_pef_days_DecFeb >= 5) & (R.n_pef_days_MarApr >= 5); okv = (R.y1_size_DecFeb >= 10) & (R.y1_size_MarApr >= 10)
for c in ['morning', 'evening', 'drop']: R[f'd_{c}_pp'] = (R[f'{c}_MarApr'] - R[f'{c}_DecFeb']).where(okp)
R['d_y1_pct'] = (100 * (R.y1_mean_MarApr / R.y1_mean_DecFeb - 1)).where(okv)
R['comparator'] = (R.night_ratio_MarApr >= 1) & (R.onshift_ratio_MarApr >= 1) & (R.n_nights_MarApr >= 5) & (R.n_shift_days_MarApr >= 5)
R['group'] = np.where(R.index == 'R21', 'R21', np.where(R.comparator, 'dirty home + shift', 'other'))
cols = ['group', 'night_ratio_DecFeb', 'night_ratio_MarApr', 'n_nights_MarApr', 'onshift_ratio_DecFeb', 'onshift_ratio_MarApr', 'own_night_MarApr_vs_DecFeb', 'own_onshift_MarApr_vs_DecFeb',
        'morning_DecFeb', 'morning_MarApr', 'd_morning_pp', 'evening_MarApr', 'd_evening_pp', 'drop_DecFeb', 'drop_MarApr', 'd_drop_pp', 'd_y1_pct', 'n_pef_days_DecFeb', 'n_pef_days_MarApr']
R[cols].round(2).to_csv(F('cases', 'tables', 'r21_haze_season.csv'))
pd.set_option('display.width', 250); print(R[cols].round(2).to_string())
print('\nmedians of change by group:'); print(R.groupby('group')[['d_morning_pp', 'd_evening_pp', 'd_drop_pp', 'd_y1_pct']].median().round(2).to_string())
