"""cases/ Part 1 (ideas 1-8, 10, 11). Exploratory, descriptive; tests only where cases/SPEC.md marks [test].
Run after case_build.py: python3 -I scripts/case_part1.py [--pm-col pm]
"""
import argparse, os, sys, math
import numpy as np, pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
from pef_stats import ols, pool  # noqa: E402
F = lambda *p: os.path.join(ROOT, *p)
ap = argparse.ArgumentParser(); ap.add_argument('--pm-col', default='pm'); A = ap.parse_args()
T = lambda name: F('cases', 'tables', name)
OUT = []
def P(*s): OUT.append(' '.join(str(x) for x in s))
TESTS = []

M = pd.read_pickle(F('cases', '_work', 'minutes.pkl'))
D = pd.read_csv(F('cases', '_work', 'days.csv'), parse_dates=['workday'])
E = pd.read_csv(F('cases', '_work', 'episodes.csv'), parse_dates=['workday'])
city = D.groupby('pid').city.first().to_dict()
tb = lambda s: s.astype(str).str.lower().isin(['true', '1', '1.0'])
V = pd.read_csv(F('foundation', 'linked_v4_pvt.csv'), low_memory=False)
V = V[V.role == 'rider'].copy(); V['excl'] = tb(V.any_exclusion); V['workday'] = pd.to_datetime(V.workday)
V = V.sort_values(['pid', 'test_start_unix']); V['order'] = V.groupby(['pid', 'workday']).cumcount() + 1
PF = pd.read_csv(F('foundation', 'linked_v4_pef.csv'), low_memory=False)
PF = PF[(PF.role == 'rider') & ~tb(PF.exact_duplicate)].copy()

def slope(d, y, x, min_n=8):
    d = d[[y, x]].dropna()
    if len(d) < min_n or d[x].std() == 0: return np.nan, np.nan, len(d)
    X = np.column_stack([np.ones(len(d)), d[x].values]); b, se, _ = ols(d[y].values.astype(float), X)
    return b[1], se[1], len(d)

def fmtpool(r, u=''):
    return f"{r['mu']:+.3f}{u} [{r['lo_hk']:+.3f}, {r['hi_hk']:+.3f}], p_HKSJ {r['p_hk']:.3f}, k {r['k']}, tau2 {r['tau2']:.3f}, share<0 {r['share_neg']:.2f}"

# ---------- Q1 ----------
P('## Q1 work on bad-air days')
rows = []
for p, g in D.groupby('pid'):
    g = g.copy(); g['z'] = (g.city_pm - g.city_pm.mean()) / g.city_pm.std()
    r = {'pid': p, 'city': city[p], 'n_days': len(g)}
    for y in ['hours_shift', 'km', 'shift_start_clock', 'return_clock']:
        b, se, n = slope(g, y, 'z'); r[f'b_{y}'] = b; r[f'se_{y}'] = se
    rows.append(r)
Q1 = pd.DataFrame(rows); Q1.to_csv(T('q1_badair_work_slopes.csv'), index=False)
for y, u in [('hours_shift', ' h'), ('km', ' km')]:
    q = Q1.dropna(subset=[f'b_{y}']); r = pool(q[f'b_{y}'], q[f'se_{y}']); TESTS.append(f'Q1 {y}')
    P(f'pooled {y} per +1 SD city PM: {fmtpool(r, u)}')
for y in ['shift_start_clock', 'return_clock']:
    q = Q1[f'b_{y}'].dropna(); P(f'{y} per-rider slopes (h per SD): median {q.median():+.2f}, IQR {q.quantile(.25):+.2f}..{q.quantile(.75):+.2f}, <0 in {(q<0).sum()}/{len(q)}')
P(Q1[['pid', 'city', 'n_days', 'b_hours_shift', 'b_km', 'b_shift_start_clock', 'b_return_clock']].round(2).to_string(index=False))

# ---------- Q2 ----------
P('\n## Q2 work hours vs city diurnal PM')
b = M[M.pm.notna()]
hh = b.groupby(['city', 'pid', 'hour_unix']).pm.agg(['mean', 'size']).reset_index(); hh = hh[hh['size'] >= 30]
ch = hh.groupby(['city', 'hour_unix'])['mean'].median().reset_index()
lt = pd.to_datetime(ch.hour_unix + 7 * 3600, unit='s'); ch['date'] = lt.dt.date; ch['hr'] = lt.dt.hour
ch['season'] = np.where(lt.dt.month.isin([12, 1, 2]), 'DecFeb', 'MarApr')
dm = ch.groupby(['city', 'date'])['mean'].agg(['mean', 'size']); ch = ch.join(dm, on=['city', 'date'], rsuffix='_d')
ch = ch[ch['size'] >= 20]; ch['rel'] = ch['mean'] / ch['mean_d']
curve = ch.groupby(['city', 'season', 'hr']).rel.median().unstack('hr'); curve.round(2).to_csv(T('q2_city_diurnal_curve.csv'))
dirty = {k: set(curve.loc[k].sort_values(ascending=False).index[:8]) for k in curve.index}
for k in curve.index: P(k, 'dirtiest 8 hours:', sorted(dirty[k]), '| max/min hour ratio', round(curve.loc[k].max() / curve.loc[k].min(), 2))
on = M[M.onshift].copy(); hr = pd.to_datetime(on.minute_unix + 7 * 3600, unit='s').dt.hour; on['hr'] = hr.values
on['season'] = np.where(on.month.isin([12, 1, 2]), 'DecFeb', 'MarApr')
on['in_dirty'] = [h in dirty[(c, s)] for c, s, h in zip(on.city, on.season, on.hr)]
bands = pd.cut(on.hr.where(on.hr >= 4, on.hr + 24), [4, 10, 14, 18, 22, 28], right=False, labels=['04-10', '10-14', '14-18', '18-22', '22-04'])
Q2 = on.groupby('pid').in_dirty.mean().rename('share_in_dirtiest_third').to_frame()
ct = pd.crosstab(on.pid, bands.astype(str), normalize='index').round(3); ct.columns = [str(c) for c in ct.columns]; Q2 = Q2.join(ct); Q2.insert(0, 'city', Q2.index.map(city))
Q2.to_csv(T('q2_work_share_dirty_hours.csv'))
P(Q2.round(2).to_string())
P('median share in dirtiest third by city:', Q2.groupby('city').share_in_dirtiest_third.median().round(2).to_dict())

# ---------- Q3 ----------
P('\n## Q3 what-if schedules')
refmap = M.drop_duplicates(['pid', 'hour_unix']).set_index(['pid', 'hour_unix']).ref
homemean = M[~M.away & M.pm.notna()].groupby(['pid', 'workday']).pm.mean()
homeall = M[~M.away & M.pm.notna()].groupby('pid').pm.mean()
daily = M[M.pm.notna()].groupby(['pid', 'workday']).pm.sum()
keep = set(zip(D.pid, D.workday))
rows = []
for (p, wd), g in M[M.onshift & M.pm.notna()].groupby(['pid', 'workday']):
    if (p, wd) not in keep or len(g) < 60: continue
    g = g.sort_values('minute_unix'); tot = g.pm.sum(); day = daily[(p, wd)]
    hm = homemean.get((p, wd), homeall[p])
    last2 = g.minute_unix > g.minute_unix.max() - 7200
    cap9 = np.arange(len(g)) >= 540
    top = g.pm.values >= np.quantile(g.pm.values, 0.9)
    r = {'pid': p, 'workday': wd}
    for nm, msk in [('end2h', last2.values), ('cap9h', cap9), ('skip_top10', top)]:
        rem = g.pm.values[msk].sum()
        r[f'{nm}_onshift_pct'] = 100 * rem / tot
        r[f'{nm}_daily_pct'] = 100 * (rem - msk.sum() * hm) / day
    gg = g[g.ref.notna()]; ratio = gg.pm / gg.ref
    for nm, sh in [('earlier2h', -7200), ('later2h', 7200)]:
        cf = refmap.reindex(list(zip(gg.pid, gg.hour_unix + sh))).values
        ok = ~np.isnan(cf)
        if ok.mean() > 0.8:
            r[f'{nm}_onshift_pct'] = 100 * ((ratio.values[ok] * cf[ok]).sum() / gg.pm.values[ok].sum() - 1)
    rows.append(r)
Q3d = pd.DataFrame(rows); Q3 = Q3d.drop(columns='workday').groupby('pid').median(); Q3.insert(0, 'city', Q3.index.map(city))
Q3.round(1).to_csv(T('q3_whatif_by_rider.csv'))
P('rider-workdays', len(Q3d)); P(Q3.round(1).to_string())
P('median of rider medians by city:'); P(Q3.groupby('city').median(numeric_only=True).round(1).to_string())

# ---------- Q4 ----------
P('\n## Q4 late nights -> next morning first PVT')
x4 = V[V.shift_crosses_0400.astype(str).str.lower() == 'true'][['pid', 'workday']].drop_duplicates()
late = set(zip(E[(E.lastmove_clock >= 24) | ((E.ret_clock >= 24) & (E.ret_clock < 29))].pid, E[(E.lastmove_clock >= 24) | ((E.ret_clock >= 24) & (E.ret_clock < 29))].workday))
late |= set(zip(x4.pid, x4.workday))
# movement after 00:00 local inside an episode whose shift began the previous workday is caught by lastmove_clock>=24;
# also count night riding minutes 00:00-04:00 (moving, onshift) as a late night for that workday
nm = M[M.onshift & M.moving & (M.clock < 4)].groupby(['pid', 'workday']).size()
late |= set(nm[nm >= 10].index)
first = V[(V.order == 1) & ~V.excl & V.y1_speed_rt100.notna()][['pid', 'workday', 'y1_speed_rt100']]
first['after_late'] = [(p, wd - pd.Timedelta(days=1)) in late for p, wd in zip(first.pid, first.workday)]
rows = []
for p, g in first.groupby('pid'):
    a = g[g.after_late]; o = g[~g.after_late]
    rows.append({'pid': p, 'city': city.get(p), 'late_days': sum(1 for q, _ in late if q == p), 'next_morning_tests': len(a),
                 'y1_diff_pct': 100 * (a.y1_speed_rt100.mean() / o.y1_speed_rt100.mean() - 1) if len(a) else np.nan})
Q4 = pd.DataFrame(rows); Q4.to_csv(T('q4_late_nights.csv'), index=False)
n8 = (Q4.next_morning_tests >= 3).sum()
P('late-night rider-workdays:', len(late), '| riders with any:', (Q4.late_days > 0).sum(), '| riders with >=3 next-morning tests:', n8, '-> test', 'run' if n8 >= 8 else 'not run (descriptive)')
P(Q4[Q4.late_days > 0].round(1).to_string(index=False))

# ---------- Q5 ----------
P('\n## Q5 trend over the study')
v = V[~V.excl].copy(); v['m30'] = v.days_since_enrolment / 30
rows = []
for p, g in v.groupby('pid'):
    r = {'pid': p, 'city': city.get(p), 'n': len(g), 'span_days': g.days_since_enrolment.max() - g.days_since_enrolment.min()}
    for y in ['y1_speed_rt100', 'med_rt', 'y3_slowest10_speed']:
        bb, se, n = slope(g, y, 'm30', 15); r[f'{y}_pct_per30d'] = 100 * bb / g[y].mean()
    bb, se, n = slope(g, 'y2_lapse_pct_500', 'm30', 15); r['lapse500_pp_per30d'] = bb
    rows.append(r)
Q5 = pd.DataFrame(rows); Q5.to_csv(T('q5_trend_by_rider.csv'), index=False)
P(Q5.round(2).to_string(index=False))
P('city medians:'); P(Q5.groupby('city').median(numeric_only=True).round(2).to_string())
P('available session columns: y1 (mean speed), med_rt, iqr_rt, max_rt, y3 slowest10 speed, lapse %, false starts; no fastest-10% column')

# ---------- Q6 ----------
P('\n## Q6 hotspots')
g6 = M[M.onshift & M.both & M.lat.notna() & (M.pm > 0)].copy()
g6['lr'] = np.log(g6.pm / g6.ref)
g6['lr_own'] = g6.lr - g6.groupby('pid').lr.transform('median')
g6['cy'] = np.floor(g6.lat / 0.01).astype(int); g6['cx'] = np.floor(g6.lon / 0.01).astype(int)
homes = pd.read_csv(F('foundation', '_work', 'home_locations.csv')); homes = homes[homes.home.astype(str) == 'True']
cells = g6.groupby(['city', 'cy', 'cx']).agg(n_min=('lr', 'size'), n_riders=('pid', 'nunique'), lr=('lr', 'mean'), lr_own=('lr_own', 'mean')).reset_index()
cells['clat'] = (cells.cy + 0.5) * 0.01; cells['clon'] = (cells.cx + 0.5) * 0.01
def near_home(lat, lon):
    dy = (homes.lat.values - lat) * 111.0; dx = (homes.lon.values - lon) * 111.0 * math.cos(math.radians(lat))
    return (np.hypot(dx, dy) < 0.3).any()
cells['near_home'] = [near_home(a, o) for a, o in zip(cells.clat, cells.clon)]
ok = cells[(cells.n_riders >= 3) & (cells.n_min >= 60) & ~cells.near_home].copy()
ok['ratio_ref'] = np.exp(ok.lr); ok['ratio_own'] = np.exp(ok.lr_own)
for c in ['BKK', 'CMI']:
    o = ok[ok.city == c]; cy0 = np.average(o.clat, weights=o.n_min); cx0 = np.average(o.clon, weights=o.n_min)
    ok.loc[o.index, 'km_n'] = ((o.clat - cy0) * 111).round(0); ok.loc[o.index, 'km_e'] = ((o.clon - cx0) * 111 * math.cos(math.radians(cy0))).round(0)
pub = ok[['city', 'km_n', 'km_e', 'n_riders', 'n_min', 'ratio_ref', 'ratio_own']].round(2)
pub.sort_values(['city', 'ratio_own'], ascending=[True, False]).to_csv(T('q6_hotspot_cells.csv'), index=False)
P('cells all:', len(cells), '| shown (>=3 riders, >=60 min, not within 300 m of a home):', len(ok), ok.groupby('city').size().to_dict(),
  '| share of on-shift minutes in shown cells:', round(ok.n_min.sum() / len(g6), 2))
for c in ['BKK', 'CMI']:
    o = pub[pub.city == c].sort_values('ratio_own')
    P(c, 'spread of cell ratio_own: p10', round(o.ratio_own.quantile(.1), 2), 'median', round(o.ratio_own.median(), 2), 'p90', round(o.ratio_own.quantile(.9), 2))
    P('top 5:\n' + o.tail(5).to_string(index=False)); P('bottom 5:\n' + o.head(5).to_string(index=False))
    P('distance from centre vs ratio_own: Spearman', round(o.ratio_own.rank().corr(np.hypot(o.km_n, o.km_e).rank()), 2))
# SVG maps (coarse, relative km, no basemap, no rider points)
def svg_map(o, c):
    W = 420; s = 12; xs = o.km_e; ys = o.km_n
    x0, x1, y0, y1 = xs.min() - 1, xs.max() + 1, ys.min() - 1, ys.max() + 1
    sc = min((W - 60) / (x1 - x0), (W - 60) / (y1 - y0))
    def col(v):
        t = max(-1, min(1, math.log(v) / math.log(1.5)))
        return f'rgb({int(255 if t > 0 else 255 * (1 + t))},{int(255 * (1 - abs(t)))},{int(255 if t < 0 else 255 * (1 - t))})'
    el = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{W + 40}" font-family="sans-serif" font-size="11">',
          f'<rect width="100%" height="100%" fill="white"/><text x="10" y="16">{c}: on-shift PM in ~1 km cells relative to each rider\'s own typical (exploratory)</text>']
    for r in o.itertuples():
        x = 30 + (r.km_e - x0) * sc; y = 30 + (y1 - r.km_n) * sc
        el.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{max(sc,2):.1f}" height="{max(sc,2):.1f}" fill="{col(r.ratio_own)}" stroke="#999" stroke-width="0.3"/>')
    el.append(f'<text x="10" y="{W + 30}">blue = cleaner than own typical (0.67x), white = 1x, red = dirtier (1.5x). Axes: km from centroid. Cells near homes removed.</text></svg>')
    open(F('cases', f'fig_q6_hotspots_{c}.svg'), 'w').write('\n'.join(el))
for c in ['BKK', 'CMI']: svg_map(pub[pub.city == c], c)

# ---------- Q7 ----------
P('\n## Q7 exposure per unit of work')
D['onh'] = D.n_onshift / 60; D['ratio_hours_on'] = D.ratio_on * D.onh; D['per_km'] = D.ratio_hours_on / D.km.where(D.km > 5)
Q7 = D.groupby('pid').agg(city=('city', 'first'), hours_on=('onh', 'median'), km=('km', 'median'), ratio_per_hour=('ratio_on', 'median'),
                          ratio_hours_day=('ratio_hours_on', 'median'), ratio_hours_per_100km=('per_km', lambda s: 100 * s.median()))
for c in ['ratio_hours_day', 'ratio_per_hour', 'ratio_hours_per_100km']:
    Q7[f'rank_{c}'] = Q7.groupby('city')[c].rank(ascending=False).astype(int)
Q7.round(2).to_csv(T('q7_dose_per_work.csv')); P(Q7.round(2).sort_values(['city', 'ratio_hours_day']).to_string())
P('Spearman (within city) dose/day vs hours:', {c: round(g.ratio_hours_day.rank().corr(g.hours_on.rank()), 2) for c, g in Q7.groupby('city')},
  '| dose/day vs ratio per hour:', {c: round(g.ratio_hours_day.rank().corr(g.ratio_per_hour.rank()), 2) for c, g in Q7.groupby('city')})

# ---------- Q8 ----------
P('\n## Q8 own sensor peak hours/months (interviewed riders)')
PR = dict(pd.read_csv(F('cases', '_work', 'p_to_r.csv')).values)  # untracked P<->R map
b = M[M.pm.notna()].copy(); b['hr'] = pd.to_datetime(b.minute_unix + 7 * 3600, unit='s').dt.hour
rows = []
for pp, r in PR.items():
    g = b[b.pid == r]; mu = g.pm.mean()
    hp = g.groupby('hr').pm.mean() / mu; hs = g[g.onshift].groupby('hr').pm.mean() / mu
    hs = hs[g[g.onshift].groupby('hr').size() >= 120]
    mp = g.groupby('month').pm.mean() / mu
    band = pd.Series({'morning 05-10': hp.loc[5:9].mean(), 'midday 10-15': hp.loc[10:14].mean(), 'afternoon 15-19': hp.loc[15:18].mean(), 'evening 19-24': hp.loc[19:23].mean(), 'night 00-05': hp.loc[0:4].mean()})
    rows.append({'P': pp, 'R': r, 'city': city[r], 'top3_hours_all': sorted(hp.sort_values().index[-3:].tolist()), 'top3_hours_onshift': sorted(hs.sort_values().index[-3:].tolist()),
                 'worst_band': band.idxmax(), 'worst_band_ratio': round(band.max(), 2), 'midday_ratio': round(band['midday 10-15'], 2), 'worst_month': int(mp.idxmax()), 'worst_month_ratio': round(mp.max(), 2)})
Q8 = pd.DataFrame(rows); Q8.to_csv(T('q8_own_peak_times.csv'), index=False); P(Q8.to_string(index=False))

# ---------- Q10 ----------
P('\n## Q10 power / sample size')
import json
pe = pd.read_csv(F('pvt_paper', 'run_raw', 'per_rider_estimates.csv')); pe = pe[(pe.model == 'F1_Y1') & pe.se_pm.notna()]
tau_v = json.load(open(F('pvt_paper', 'run_raw', 'pooled.json')))['pooled']['F1_Y1']['pm']['tau2']
pp_ = pd.read_csv(F('peakflow_paper', 'run_raw', 'per_person.csv')); pp_ = pp_[(pp_.model == 'L1') & pp_.se0.notna()]
tau_p = json.load(open(F('peakflow_paper', 'run_raw', 'pooled.json')))['models']['L1']['all:PM_sd:z']['tau2']
rows = []
for nm, se, ndays, tau, bound in [('PVT Y1 % per SD', pe.se_pm.values, pe.n_days.values, tau_v, 1.01), ('PEF evening pp per SD', pp_.se0.values, pp_.n.values, tau_p, 1.68)]:
    k = len(se); se_now = math.sqrt(1 / (1 / (se ** 2 + tau)).sum()); s2 = np.median(se ** 2 * ndays); dmed = np.median(ndays)
    r = {'outcome': nm, 'k': k, 'median_days': dmed, 'tau2': round(tau, 3), 'within_var_x_days': round(s2, 1), 'MDE80_now': round(2.8 * se_now, 2), 'bound': bound}
    for frac in [0.5, 0.25]:
        tgt = frac * bound
        for d in [30, 60, 120]:
            v1 = tau + s2 / d; kneed = math.ceil(v1 * (2.8 / tgt) ** 2); r[f'riders_for_{frac}x_at_{d}d'] = kneed
    r['min_riders_0.5x_infinite_days'] = math.ceil(tau * (2.8 / (0.5 * bound)) ** 2); r['min_riders_0.25x_infinite_days'] = math.ceil(tau * (2.8 / (0.25 * bound)) ** 2)
    rows.append(r)
Q10 = pd.DataFrame(rows).T; Q10.to_csv(T('q10_power.csv')); P(Q10.to_string())

# ---------- Q11 ----------
P('\n## Q11 four months of self-testing')
pf = PF.copy(); pf['ld'] = (pf.peak_flow.round() % 10).astype(int)
pf = pf.sort_values(['pid', 'reading_unix']); pf['same_prev'] = pf.peak_flow.eq(pf.groupby('pid').peak_flow.shift())
v = V.copy(); v['p7'] = tb(v.P7_interruption).astype(float)
rows = []
for p in sorted(city):
    a = pf[pf.pid == p]; q = v[v.pid == p]; mo = a[a.period == 'morning']; ev = a[a.period == 'evening']; f1 = q[q.order == 1]
    rows.append({'pid': p, 'city': city[p], 'n_pef': len(a), 'pef_last_digit_0_or_5': (a.ld % 5 == 0).mean(), 'pef_last_digit_0': (a.ld == 0).mean(), 'pef_same_as_previous': a.same_prev.mean(),
                 'morning_before_shift': (mo.protocol_slot == 'before_shift').mean(), 'evening_after_home': (ev.protocol_slot == 'after_home').mean(),
                 'pvt1_before_shift': (f1.protocol_slot == 'before_shift').mean(), 'p7_before': q[q.protocol_slot == 'before_shift'].p7.mean(), 'p7_during': q[q.protocol_slot == 'during_shift'].p7.mean(),
                 'tapgap_med_before': q[q.protocol_slot == 'before_shift'].max_tap_gap_s.median(), 'tapgap_med_during': q[q.protocol_slot == 'during_shift'].max_tap_gap_s.median()})
Q11 = pd.DataFrame(rows); Q11.to_csv(T('q11_selftesting_by_rider.csv'), index=False); P(Q11.round(2).to_string(index=False))
bs, ss = [], []
for p, g in v[v.protocol_slot.isin(['before_shift', 'during_shift'])].groupby('pid'):
    g = g.assign(dur=(g.protocol_slot == 'during_shift').astype(float))
    if g.dur.sum() >= 3 and (1 - g.dur).sum() >= 3 and g.p7.std() > 0:
        bb, se, n = slope(g, 'p7', 'dur', 10)
        if np.isfinite(se) and se > 0: bs.append(bb * 100); ss.append(se * 100)
r = pool(bs, ss); TESTS.append('Q11 P7 during vs before'); P('[test] P7 interruption, during_shift - before_shift (pp):', fmtpool(r, ' pp'), '(riders with variation:', len(bs), ')')
P('overall P7 rate before/during/after_home:', v.groupby('protocol_slot').p7.mean().round(3).to_dict())
# adherence by month
v['month'] = pd.to_datetime(v.timestamp_end_local).dt.to_period('M')
ad = v.groupby(['pid', 'month', 'workday']).size().reset_index(name='n')
adm = ad.groupby('month').agg(rider_days=('n', 'size'), sessions_per_day=('n', 'mean'), share_days_3plus=('n', lambda s: (s >= 3).mean()))
pf['month'] = pd.to_datetime(pf.timestamp_local).dt.to_period('M')
pd_ = pf.groupby(['pid', 'month', 'workday']).period.agg(lambda s: {'morning', 'evening'} <= set(s)).reset_index(name='both')
adm = adm.join(pd_.groupby('month').both.mean().rename('pef_share_days_both'))
adm.round(2).to_csv(T('q11_adherence_by_month.csv')); P(adm.round(2).to_string())
P('pooled last digit: share 0', round((pf.ld == 0).mean(), 3), 'share 5', round((pf.ld == 5).mean(), 3), '| same as previous', round(pf.same_prev.mean(), 3))

P('\nInferential tests in Part 1:', len(TESTS), TESTS)
open(F('cases', '_work', 'part1_digest.txt'), 'w').write('\n'.join(OUT))
print('\n'.join(OUT))
