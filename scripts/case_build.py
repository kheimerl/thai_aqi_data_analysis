"""cases/ build step (exploratory, descriptive). See cases/SPEC.md (2026-11-13).

Builds rider minute table with city reference, shift/away/night flags, and a rider-workday table.
numpy/pandas only. Run: python3 -I scripts/case_build.py [--pm-col pm]
"""
import argparse, os
import numpy as np, pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
F = lambda *p: os.path.join(ROOT, *p)
ap = argparse.ArgumentParser(); ap.add_argument('--pm-col', default='pm'); A = ap.parse_args()
os.makedirs(F('cases', '_work'), exist_ok=True); os.makedirs(F('cases', 'tables'), exist_ok=True)

coh = pd.read_csv(F('foundation', 'cohort.csv'))
riders = coh[coh.role == 'rider'].copy()
city = dict(zip(riders.pid, riders.city))
win = {r.pid: (pd.Timestamp(r.enrol_start).date(), pd.Timestamp(r.enrol_end).date()) for r in riders.itertuples()}

def wd_of(unix):
    return pd.to_datetime(np.asarray(unix, 'int64') + 7 * 3600 - 4 * 3600, unit='s').normalize()

frames = []
for node in list(range(6011, 6031)) + [6032]:
    m = pd.read_csv(F('foundation', '_work', f'minute_{node}.csv'),
                    usecols=['minute_unix', A.pm_col, 'lat', 'lon', 'speed_kmh', 'km_step'])
    m = m.rename(columns={A.pm_col: 'pm'})
    m['workday'] = wd_of(m.minute_unix)
    d = m.workday.dt.date
    if node == 6021:
        m['pid'] = np.where(d <= pd.Timestamp('2026-02-26').date(), 'R11', np.where(d >= pd.Timestamp('2026-03-04').date(), 'R12', None))
    elif node == 6022:
        m['pid'] = np.where(d <= pd.Timestamp('2026-03-25').date(), 'R13', np.where(d >= pd.Timestamp('2026-03-27').date(), 'R14', None))
    elif node in (6029, 6032):
        m['pid'] = 'R22'
    else:
        m['pid'] = riders.loc[riders.sensor.astype(str) == str(node), 'pid'].iloc[0]
    m = m[m.pid.notna()]
    m = m[[win[p][0] <= x <= win[p][1] for p, x in zip(m.pid, m.workday.dt.date)]]
    frames.append(m)
M = pd.concat(frames, ignore_index=True)
M['city'] = M.pid.map(city)
M['hour_unix'] = (M.minute_unix // 3600) * 3600
loc = pd.to_datetime(M.minute_unix + 7 * 3600, unit='s')
M['clock'] = loc.dt.hour + loc.dt.minute / 60
M['month'] = loc.dt.month

# city reference: median of OTHER riders' hourly means (>= 30 PM minutes), >= 2 others
H = M[M.pm.notna()].groupby(['pid', 'hour_unix']).pm.agg(['mean', 'size']).reset_index()
H = H[H['size'] >= 30]
refs = []
for c in ['BKK', 'CMI']:
    P = H[H.pid.map(city) == c].pivot(index='hour_unix', columns='pid', values='mean')
    for p in P.columns:
        o = P.drop(columns=p)
        r = o.median(axis=1).where(o.notna().sum(axis=1) >= 2)
        refs.append(pd.DataFrame({'pid': p, 'hour_unix': r.index, 'ref': r.values}))
R = pd.concat(refs).dropna()
M = M.merge(R, on=['pid', 'hour_unix'], how='left')

# episodes
E = pd.read_csv(F('foundation', '_work', 'shift_episodes.csv'))
E = E[E.pid.isin(city)]
E['away_end'] = E.home_return.fillna(E.block_end)
E['workday'] = wd_of(E.shift_start.astype('int64'))
M['onshift'] = False; M['away'] = False
for p, g in E.groupby('pid'):
    idx = M.index[M.pid == p]; t = M.loc[idx, 'minute_unix'].values
    on = np.zeros(len(t), bool); aw = np.zeros(len(t), bool)
    for e in g.itertuples():
        on |= (t >= e.shift_start) & (t <= e.last_move)
        aw |= (t >= e.away_start) & (t < e.away_end)
    M.loc[idx, 'onshift'] = on; M.loc[idx, 'away'] = aw
M['moving'] = M.speed_kmh > 5
M['night_home'] = (M.clock < 5) & ~M.away
M['both'] = M.pm.notna() & M.ref.notna()
M.to_pickle(F('cases', '_work', 'minutes.pkl'))

# rider-workday table
b = M[M.both]
g = b.groupby(['pid', 'workday'])
D = pd.DataFrame({'pm_sum': g.pm.sum(), 'ref_sum': g.ref.sum(), 'n_both': g.size()})
D['n_pm'] = M[M.pm.notna()].groupby(['pid', 'workday']).size()
D['pm_mean'] = M[M.pm.notna()].groupby(['pid', 'workday']).pm.mean()
for flag in ['onshift', 'night_home']:
    gg = b[b[flag]].groupby(['pid', 'workday'])
    D[f'pm_sum_{flag}'] = gg.pm.sum(); D[f'ref_sum_{flag}'] = gg.ref.sum(); D[f'n_{flag}'] = gg.size()
ev = b[(b.clock >= 17) & (b.clock < 23)].groupby(['pid', 'workday'])
D['pm_sum_eve'] = ev.pm.sum(); D['ref_sum_eve'] = ev.ref.sum()
cl = b.assign(clean=b.pm < b.ref)
D['min_clean_24h'] = cl.groupby(['pid', 'workday']).clean.sum()
D['min_clean_onshift'] = cl[cl.onshift].groupby(['pid', 'workday']).clean.sum()
D['km'] = M[M.onshift & M.moving].groupby(['pid', 'workday']).km_step.sum()
D = D.reset_index()

E['span_h'] = (E.last_move - E.shift_start) / 3600
E['start_clock'] = (E.shift_start + 3 * 3600) % 86400 / 3600 + 4  # hours since 00:00 of workday, 4..28
# ret_clock: hours after 00:00 local of the workday (local midnight unix = workday - 7 h)
wd_unix_local0 = (E.workday.astype('int64') // 10**9) - 7 * 3600
E['ret_clock'] = np.where(E.home_return.notna(), (E.home_return - wd_unix_local0) / 3600, np.nan)
E['lastmove_clock'] = (E.last_move - wd_unix_local0) / 3600
E['out_h'] = np.where(E.home_return.notna() & ~E.start_censored.astype(bool), (E.home_return - E.away_start) / 3600, np.nan)
ge = E.groupby(['pid', 'workday'])
W = pd.DataFrame({'hours_shift': ge.span_h.sum(), 'shift_start_clock': (ge.shift_start.min() - ge.workday.first().astype('int64') // 10**9 + 7 * 3600 - 7 * 3600) / 3600,
                  'return_clock': ge.ret_clock.max(), 'lastmove_clock': ge.lastmove_clock.max(), 'out_h': ge.out_h.sum(min_count=1),
                  'late_night': ge.lastmove_clock.max() >= 24}).reset_index()
wd0 = (W.workday.astype('int64') // 10**9) - 7 * 3600
W['shift_start_clock'] = (ge.shift_start.min().values - wd0.values) / 3600
D = D.merge(W, on=['pid', 'workday'], how='inner')
D = D[D.n_pm >= 600]
D['city'] = D.pid.map(city)
D['ratio24'] = D.pm_sum / D.ref_sum
D['ratio_on'] = D.pm_sum_onshift / D.ref_sum_onshift
D['ratio_night'] = D.pm_sum_night_home / D.ref_sum_night_home
D['ratio_eve'] = D.pm_sum_eve / D.ref_sum_eve
# city PM that workday from other riders (24 h workday means)
allday = M[M.pm.notna()].groupby(['pid', 'workday']).pm.agg(['mean', 'size']).reset_index()
allday = allday[allday['size'] >= 600]; allday['city'] = allday.pid.map(city)
cp = []
for r in D.itertuples():
    o = allday[(allday.city == r.city) & (allday.workday == r.workday) & (allday.pid != r.pid)]
    cp.append(o['mean'].mean() if len(o) >= 2 else np.nan)
D['city_pm'] = cp
D.to_csv(F('cases', '_work', 'days.csv'), index=False)
E.to_csv(F('cases', '_work', 'episodes.csv'), index=False)
print('minutes', len(M), 'with ref', int(M.both.sum()), '| rider-workdays', len(D), 'riders', D.pid.nunique())
print(D.groupby('pid').size().to_string())
