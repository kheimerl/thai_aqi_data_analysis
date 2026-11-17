"""cases/ S9: three case studies (purifier user, rider with a cold, night worker). See cases/SPEC.md S9 addendum (2026-11-16).
Exploratory, descriptive, n = 1 per case, raw PM -> ratios only. Run after case_build.py: python3 -I scripts/case_s9.py [--pm-col pm]
"""
import argparse, os, math
import numpy as np, pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
F = lambda *p: os.path.join(ROOT, *p)
ap = argparse.ArgumentParser(); ap.add_argument('--pm-col', default='pm'); A = ap.parse_args()  # pm column chosen in case_build.py
T = lambda n: F('cases', 'tables', n)
OUT = []
def P(*s): OUT.append(' '.join(str(x) for x in s))
tb = lambda s: s.astype(str).str.lower().isin(['true', '1', '1.0'])
RNG = np.random.default_rng(2026)

M = pd.read_pickle(F('cases', '_work', 'minutes.pkl'))
M['date'] = pd.to_datetime(M.minute_unix + 7 * 3600, unit='s').dt.normalize()
M['hr'] = M.clock.astype(int)
D = pd.read_csv(F('cases', '_work', 'days.csv'), parse_dates=['workday'])
b = M[M.both]

def boot(g, key, min_n):
    s = g.groupby(key).agg(pm=('pm', 'sum'), ref=('ref', 'sum'), n=('pm', 'size')); s = s[s.n >= min_n]
    if len(s) < 3: return np.nan, np.nan, np.nan, len(s), s
    est = s.pm.sum() / s.ref.sum(); idx = RNG.integers(0, len(s), (2000, len(s)))
    bs = s.pm.values[idx].sum(1) / s.ref.values[idx].sum(1)
    return est, np.quantile(bs, .025), np.quantile(bs, .975), len(s), s
fr = lambda e: f'{e[0]:.2f} [{e[1]:.2f}, {e[2]:.2f}] (n {e[3]})'

# ---------------- Case 1 ----------------
P('## Case 1 purifier user R21 vs R18, R22 (and all CMI for offset check)')
night = b[b.night_home]; eve = b[(b.clock >= 18) & ~b.away]; on = b[b.onshift]
cmi = sorted(D[D.city == 'CMI'].pid.unique()); rows = []; series = {}
for p in cmi:
    n_ = boot(night[night.pid == p], 'date', 120); e_ = boot(eve[eve.pid == p], 'workday', 60); o_ = boot(on[on.pid == p], 'workday', 60)
    rows.append({'pid': p, 'night_home': n_[0], 'night_lo': n_[1], 'night_hi': n_[2], 'n_nights': n_[3], 'eve_home': e_[0], 'eve_lo': e_[1], 'eve_hi': e_[2], 'n_eve': e_[3],
                 'onshift': o_[0], 'on_lo': o_[1], 'on_hi': o_[2], 'n_on': o_[3], 'home_div_onshift': n_[0] / o_[0]})
    series[p] = n_[4]
C1 = pd.DataFrame(rows).set_index('pid'); C1['rank_home_div_on'] = C1.home_div_onshift.rank().astype(int)  # 1 = cleanest home relative to own road
C1.round(3).to_csv(T('s9_case1_purifier_all_cmi.csv'))
P(C1[['night_home', 'night_lo', 'night_hi', 'n_nights', 'eve_home', 'eve_lo', 'eve_hi', 'onshift', 'on_lo', 'on_hi', 'home_div_onshift', 'rank_home_div_on']].round(2).to_string())
for p in ['R21', 'R18', 'R22']:
    g = night[night.pid == p]; s = series[p]
    mo = s.groupby(s.index.month).apply(lambda x: x.pm.sum() / x.ref.sum()); cnt = s.groupby(s.index.month).size()
    hh = g.groupby('hr').apply(lambda x: x.pm.sum() / x.ref.sum())
    P(p, 'night ratio by month:', {int(k): f'{v:.2f} (n {cnt[k]})' for k, v in mo.items()}, '| by hour:', {int(k): round(v, 2) for k, v in hh.items()})
    s2 = (s.pm / s.ref); P(p, 'per-night ratio: share of nights < 0.5', round((s2 < .5).mean(), 2), '| first 10 vs last 10 nights median', round(s2.iloc[:10].median(), 2), round(s2.iloc[-10:].median(), 2))
    pd.DataFrame({'pid': p, 'date': s.index, 'night_ratio': s2.values, 'n_min': s.n.values}).to_csv(T(f's9_case1_nights_{p}.csv'), index=False)

# ---------------- Case 2 ----------------
P('\n## Case 2 rider with a cold R16')
PF = pd.read_csv(F('foundation', 'linked_v4_pef.csv'), low_memory=False)
PF = PF[(PF.pid == 'R16') & ~tb(PF.exact_duplicate) & ~tb(PF.any_exclusion)].copy(); PF['date'] = pd.to_datetime(PF.timestamp_local).dt.normalize()
pf = PF.sort_values('peak_flow').groupby(['workday', 'period']).last().reset_index().sort_values('reading_unix')
V = pd.read_csv(F('foundation', 'linked_v4_pvt.csv'), low_memory=False)
V = V[(V.pid == 'R16') & ~tb(V.any_exclusion) & V.y1_speed_rt100.notna()]
y = V.groupby('workday').y1_speed_rt100.mean(); y = 100 * y / y.median(); y.index = pd.to_datetime(y.index)
pf['workday'] = pd.to_datetime(pf.workday)
day = pf.pivot_table(index='workday', columns='period', values=['pct_pb', 'peak_flow']); day.columns = [f'{a}_{b_}' for a, b_ in day.columns]
day = day.join(y.rename('y1_pct_own_median'), how='outer')
d16 = D[D.pid == 'R16'].set_index('workday'); own_med = d16.pm_mean.median()
day = day.join((d16.pm_mean / own_med).rename('pm_vs_own_median')).join(d16.ratio24.rename('ratio_city'))
W0, W1 = pd.Timestamp('2026-03-26'), pd.Timestamp('2026-04-02')
day['illness_window'] = (day.index >= W0) & (day.index <= W1); day.round(2).to_csv(T('s9_case2_r16_daily.csv'))
P(day[(day.index >= '2026-03-05') & (day.index <= '2026-04-15')].round(1).to_string())
for per in ['morning', 'evening']:
    q = pf[pf.period == per].set_index('workday').pct_pb
    win = q[(q.index >= W0) & (q.index <= W1)]
    means = [q[(q.index >= s_) & (q.index < s_ + pd.Timedelta(days=8))] for s_ in pd.date_range(q.index.min(), q.index.max())]
    means = np.array([m.mean() for m in means if len(m) >= 2])
    P(per, 'window entries', win.round(1).tolist(), 'mean', round(win.mean(), 1), '| own median', round(q.median(), 1),
      '| percentile among', len(means), '8-day windows:', round(100 * (means < win.mean()).mean(), 1))
thr = pf.pct_pb.quantile(.25); low = (pf.pct_pb < thr).values; runs = []; i = 0
while i < len(low):
    if low[i]:
        j = i
        while j + 1 < len(low) and low[j + 1]: j += 1
        if j - i + 1 >= 3: runs.append((pf.workday.iloc[i].date(), pf.workday.iloc[j].date(), j - i + 1, round(pf.pct_pb.iloc[i:j + 1].mean(), 1)))
        i = j + 1
    else: i += 1
P('low = %PB <', round(thr, 1), '| runs of >=3 consecutive low entries:', runs)
for nm, a, z in [('Mar12-25', '2026-03-12', '2026-03-25'), ('Mar26-Apr1', '2026-03-26', '2026-04-01')]:
    w = day[(day.index >= a) & (day.index <= z)]
    P('exposure', nm, ': pm vs own median (median of days)', round(w.pm_vs_own_median.median(), 2), 'city ratio', round(w.ratio_city.median(), 2), 'days', int(w.pm_vs_own_median.notna().sum()))
rest = day[(day.index < '2026-03-12') | (day.index > '2026-04-01')]
P('exposure rest of study: pm vs own median', round(rest.pm_vs_own_median.median(), 2), 'city ratio', round(rest.ratio_city.median(), 2), 'days', int(rest.pm_vs_own_median.notna().sum()))
P('PVT Y1 % of own median, window days:', day[day.illness_window].y1_pct_own_median.dropna().round(1).tolist(), '| own percentile of window mean:',
  round(100 * (y < day[day.illness_window].y1_pct_own_median.mean()).mean(), 1))

# ---------------- Case 3 ----------------
P('\n## Case 3 night worker R19')
o19 = on[on.pid == 'R19']; allon = M[M.onshift & (M.pid == 'R19')]
share = allon.groupby('hr').size() / len(allon)
curve = pd.read_csv(T('q2_city_diurnal_curve.csv')).set_index(['city', 'season']).loc['CMI']; curve.columns = curve.columns.astype(int)
hr_ratio = o19.groupby('hr').apply(lambda x: x.pm.sum() / x.ref.sum()); hr_n = o19.groupby('hr').size()
C3 = pd.DataFrame({'share_of_work': share, 'onshift_ratio_to_ref': hr_ratio, 'onshift_minutes': hr_n,
                   'city_curve_DecFeb': curve.loc['DecFeb'], 'city_curve_MarApr': curve.loc['MarApr']}).fillna({'share_of_work': 0})
# (a) vs other riders on shift in the same hour
oc = M[M.onshift & M.pm.notna() & (M.city == 'CMI')].groupby(['pid', 'hour_unix']).pm.agg(['mean', 'size']).reset_index()
oc = oc[oc['size'] >= 15]; mine = oc[oc.pid == 'R19'].set_index('hour_unix'); oth = oc[oc.pid != 'R19']
om = oth.groupby('hour_unix')['mean'].median(); on_ = oth.groupby('hour_unix').size()
j = mine.join(om.rename('oth'), how='inner'); j = j.join(on_.rename('n_oth'))
j['hr'] = pd.to_datetime(j.index + 7 * 3600, unit='s').hour; j['lr'] = np.log(j['mean'] / j.oth)
C3['vs_others_onshift_ratio'] = j.groupby('hr').lr.mean().pipe(np.exp); C3['vs_others_hours'] = j.groupby('hr').size()
C3.round(3).to_csv(T('s9_case3_r19_by_hour.csv')); P(C3.round(2).to_string())
bsj = np.exp(j.lr.values[RNG.integers(0, len(j), (2000, len(j)))].mean(1))
P('(a) R19 on-shift vs other riders on shift, same hour: geometric mean', round(math.exp(j.lr.mean()), 2), f'[{np.quantile(bsj,.025):.2f}, {np.quantile(bsj,.975):.2f}]',
  '| hours', len(j), 'of', len(mine), 'R19 on-shift hours; clock hours covered', sorted(j.hr.unique().tolist()), '| median other riders per hour', on_.reindex(j.index).median())
P('    by night (20-04) vs day hours:', {k: round(math.exp(g.lr.mean()), 2) for k, g in j.groupby(np.where((j.hr >= 20) | (j.hr < 4), 'night', 'day'))}, j.groupby(np.where((j.hr >= 20) | (j.hr < 4), 'night', 'day')).size().to_dict())
home = M[(M.pid == 'R19') & ~M.away & M.pm.notna()].groupby('workday').pm.agg(['sum', 'size']); onw = M[(M.pid == 'R19') & M.onshift & M.pm.notna()].groupby('workday').pm.agg(['sum', 'size'])
hw = home.join(onw, lsuffix='_h', rsuffix='_o', how='inner'); hw = hw[(hw.size_h >= 60) & (hw.size_o >= 60)]
r = (hw.sum_o / hw.size_o) / (hw.sum_h / hw.size_h); idx = RNG.integers(0, len(hw), (2000, len(hw)))
bs = (hw.sum_o.values[idx].sum(1) / hw.size_o.values[idx].sum(1)) / (hw.sum_h.values[idx].sum(1) / hw.size_h.values[idx].sum(1))
P('(b) R19 on-shift mean / own at-home mean, same workday: pooled', round((hw.sum_o.sum() / hw.size_o.sum()) / (hw.sum_h.sum() / hw.size_h.sum()), 2),
  f'[{np.quantile(bs,.025):.2f}, {np.quantile(bs,.975):.2f}] median per day', round(r.median(), 2), 'n', len(hw))
# same (b) for other CMI riders, for context
ctx = {}
for p in [x for x in cmi if x != 'R19']:
    h_ = M[(M.pid == p) & ~M.away & M.pm.notna()].groupby('workday').pm.agg(['sum', 'size']); o_ = M[(M.pid == p) & M.onshift & M.pm.notna()].groupby('workday').pm.agg(['sum', 'size'])
    w_ = h_.join(o_, lsuffix='_h', rsuffix='_o', how='inner'); w_ = w_[(w_.size_h >= 60) & (w_.size_o >= 60)]
    if len(w_): ctx[p] = round((w_.sum_o.sum() / w_.size_o.sum()) / (w_.sum_h.sum() / w_.size_h.sum()), 2)
P('    other CMI riders on-shift / own home:', ctx, '| median', np.median(list(ctx.values())))
V19 = pd.read_csv(F('foundation', 'linked_v4_pvt.csv'), low_memory=False); V19 = V19[(V19.pid == 'R19') & ~tb(V19.any_exclusion)].sort_values('test_start_unix')
E = pd.read_csv(F('cases', '_work', 'episodes.csv'), parse_dates=['workday']); ss = E[E.pid == 'R19'].groupby('workday').shift_start.min()
V19['workday'] = pd.to_datetime(V19.workday); V19['ss'] = V19.workday.map(ss)
P('R19 PVT clock hours (count):', V19.clock_hour.astype(float).astype(int).value_counts().sort_index().to_dict())
pr = []
for wd, g in V19.groupby('workday'):
    if len(g) >= 2 and pd.notna(g.ss.iloc[0]) and g.test_start_unix.iloc[-1] >= g.ss.iloc[0] + 3 * 3600:
        pr.append(100 * (g.y1_speed_rt100.iloc[-1] / g.y1_speed_rt100.iloc[0] - 1))
pr = np.array(pr); bsp = np.median(pr[RNG.integers(0, len(pr), (2000, len(pr)))], 1)
P('PVT last vs first of workday (last >= 3 h after shift start): median', round(np.median(pr), 1), f'% [{np.quantile(bsp,.025):.1f}, {np.quantile(bsp,.975):.1f}] n', len(pr), '| share slower', round((pr < 0).mean(), 2))
F19 = pd.read_csv(F('foundation', 'linked_v4_pef.csv'), low_memory=False); F19 = F19[(F19.pid == 'R19') & ~tb(F19.exact_duplicate) & ~tb(F19.any_exclusion)].sort_values('reading_unix')
P('R19 PEF clock hours (count):', F19.clock_hour.astype(float).astype(int).value_counts().sort_index().to_dict(), '| periods', F19.period.value_counts().to_dict())
pe = [g.pct_pb.iloc[-1] - g.pct_pb.iloc[0] for _, g in F19.groupby('workday') if len(g) >= 2]
pe = np.array(pe); bse = np.median(pe[RNG.integers(0, len(pe), (2000, len(pe)))], 1)
P('PEF last - first entry of workday (pp): median', round(np.median(pe), 2), f'[{np.quantile(bse,.025):.2f}, {np.quantile(bse,.975):.2f}] n', len(pe), '| mean', round(pe.mean(), 2))

# ---------------- SVGs ----------------
def axes(W, H, title, xl):
    return [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" font-family="sans-serif" font-size="10"><rect width="100%" height="100%" fill="white"/>',
            f'<text x="8" y="14" font-size="11">{title}</text>', f'<text x="50" y="{H-6}">{xl}</text>']
t0, t1 = pd.Timestamp('2025-12-10'), pd.Timestamp('2026-05-01')
W, H = 640, 300; sx = lambda d: 50 + (d - t0) / (t1 - t0) * (W - 70)
ly = lambda v: 30 + (math.log(1 / 0.08) - math.log(max(min(v, 8), 0.08))) / math.log(100) * (H - 70)
el = axes(W, H, 'Case 1: night-at-home PM / concurrent city reference, per night (log scale; exploratory, raw PM)', 'date (Dec 2025 - Apr 2026)')
for v in [0.1, 0.5, 1, 2, 5]: el.append(f'<line x1="50" x2="{W-20}" y1="{ly(v):.1f}" y2="{ly(v):.1f}" stroke="#ddd"/><text x="22" y="{ly(v)+3:.1f}">{v}</text>')
cols = {'R21': '#2a9d8f', 'R18': '#e76f51', 'R22': '#6d597a'}
for p, c in cols.items():
    s = series[p]
    for d_, rr in zip(s.index, s.pm / s.ref): el.append(f'<circle cx="{sx(d_):.1f}" cy="{ly(rr):.1f}" r="2.5" fill="{c}" opacity="0.8"/>')
el.append('<text x="60" y="290" fill="#2a9d8f">R21/P2 purifier</text><text x="170" y="290" fill="#e76f51">R18/P10 AC</text><text x="260" y="290" fill="#6d597a">R22/P11 AC + purifier</text></svg>')
open(F('cases', 'fig_s9_case1_night_home.svg'), 'w').write('\n'.join(el))
t0, t1 = pd.Timestamp('2025-12-05'), pd.Timestamp('2026-05-01')
sy = lambda v: 30 + (115 - min(max(v, 70), 115)) / 45 * (H - 70)
el = axes(W, H, 'Case 2: R16/P4 morning and evening %PB and PVT speed (% of own median); shaded = reported cold window', 'date')
el.append(f'<rect x="{sx(W0):.1f}" y="30" width="{sx(W1)-sx(W0)+2:.1f}" height="{H-70}" fill="#f6d6d6"/>')
for v in [80, 90, 100, 110]: el.append(f'<line x1="50" x2="{W-20}" y1="{sy(v):.1f}" y2="{sy(v):.1f}" stroke="#ddd"/><text x="25" y="{sy(v)+3:.1f}">{v}</text>')
for c_, colr, sh in [('pct_pb_morning', '#1d3557', 'circle'), ('pct_pb_evening', '#e63946', 'circle'), ('y1_pct_own_median', '#8d99ae', 'rect')]:
    for d_, v in day[c_].dropna().items():
        el.append(f'<circle cx="{sx(d_):.1f}" cy="{sy(v):.1f}" r="2.8" fill="{colr}"/>' if sh == 'circle' else f'<rect x="{sx(d_)-2:.1f}" y="{sy(v)-2:.1f}" width="4" height="4" fill="{colr}"/>')
el.append('<text x="60" y="290" fill="#1d3557">morning %PB</text><text x="150" y="290" fill="#e63946">evening %PB</text><text x="240" y="290" fill="#8d99ae">PVT speed % own median</text></svg>')
open(F('cases', 'fig_s9_case2_r16.svg'), 'w').write('\n'.join(el))
order = list(range(12, 24)) + list(range(0, 12)); bx = lambda i: 50 + i * (W - 70) / 24
el = axes(W, H, 'Case 3: R19 work time by clock hour (bars), CMI diurnal curve (lines), R19 on-shift ratio to reference (dots)', 'clock hour (12:00 -> 11:00)')
yb = lambda v: H - 40 - v / 0.2 * (H - 80); yr = lambda v: H - 40 - min(v, 2.2) / 2.2 * (H - 80)
for i, h in enumerate(order):
    el.append(f'<rect x="{bx(i)+1:.1f}" y="{yb(C3.share_of_work.get(h,0)):.1f}" width="{(W-70)/24-2:.1f}" height="{H-40-yb(C3.share_of_work.get(h,0)):.1f}" fill="#cfd8dc"/><text x="{bx(i)+3:.1f}" y="{H-28}">{h}</text>')
for c_, colr in [('city_curve_DecFeb', '#457b9d'), ('city_curve_MarApr', '#e9c46a')]:
    pts = ' '.join(f'{bx(i)+(W-70)/48:.1f},{yr(C3[c_].get(h)):.1f}' for i, h in enumerate(order) if pd.notna(C3[c_].get(h)))
    el.append(f'<polyline points="{pts}" fill="none" stroke="{colr}" stroke-width="2"/>')
for i, h in enumerate(order):
    v = C3.onshift_ratio_to_ref.get(h)
    if pd.notna(v) and C3.onshift_minutes.get(h, 0) >= 120: el.append(f'<circle cx="{bx(i)+(W-70)/48:.1f}" cy="{yr(v):.1f}" r="3.5" fill="#d62828"/>')
el.append(f'<line x1="50" x2="{W-20}" y1="{yr(1):.1f}" y2="{yr(1):.1f}" stroke="#999" stroke-dasharray="3,3"/><text x="{W-60}" y="{yr(1)-3:.1f}">ratio 1</text>')
el.append('<text x="60" y="290" fill="#457b9d">city curve Dec-Feb</text><text x="170" y="290" fill="#e9c46a">Mar-Apr</text><text x="230" y="290" fill="#d62828">R19 on-shift / reference (hours with &#8805;120 min)</text></svg>')
open(F('cases', 'fig_s9_case3_r19.svg'), 'w').write('\n'.join(el))

open(F('cases', '_work', 's9_digest.txt'), 'w').write('\n'.join(OUT)); print('\n'.join(OUT))
