"""cases/ rider profile + Part 2 case questions (A1-D16). Exploratory, descriptive, no tests.
Run after case_build.py and case_part1.py: python3 -I scripts/case_part2.py [--pm-col pm]
"""
import argparse, os, sys, math
import numpy as np, pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
from pef_stats import ols  # noqa: E402
F = lambda *p: os.path.join(ROOT, *p)
ap = argparse.ArgumentParser(); ap.add_argument('--pm-col', default='pm'); A = ap.parse_args()
T = lambda name: F('cases', 'tables', name)
OUT = []
def P(*s): OUT.append(' '.join(str(x) for x in s))
tb = lambda s: s.astype(str).str.lower().isin(['true', '1', '1.0'])

M = pd.read_pickle(F('cases', '_work', 'minutes.pkl'))
D = pd.read_csv(F('cases', '_work', 'days.csv'), parse_dates=['workday'])
city = D.groupby('pid').city.first().to_dict()
V = pd.read_csv(F('foundation', 'linked_v4_pvt.csv'), low_memory=False)
V = V[(V.role == 'rider')].copy(); V['excl'] = tb(V.any_exclusion); V['workday'] = pd.to_datetime(V.workday)
V = V.sort_values(['pid', 'test_start_unix']); V['order'] = V.groupby(['pid', 'workday']).cumcount() + 1
PF = pd.read_csv(F('foundation', 'linked_v4_pef.csv'), low_memory=False)
PF = PF[(PF.role == 'rider') & ~tb(PF.exact_duplicate) & ~tb(PF.any_exclusion)].copy(); PF['workday'] = pd.to_datetime(PF.workday)
pp = pd.read_csv(F('peakflow_paper', 'run_drop', 'per_rider_pp.csv')).set_index('pid')
pe = pd.read_csv(F('pvt_paper', 'run_raw', 'per_rider_estimates.csv')); pe = pe[pe.model == 'F2a'].set_index('pid')
Q1 = pd.read_csv(T('q1_badair_work_slopes.csv')).set_index('pid')
Q2 = pd.read_csv(T('q2_work_share_dirty_hours.csv')).set_index('pid')
PR = dict(pd.read_csv(F('cases', '_work', 'p_to_r.csv')).values)  # untracked P<->R map
RP = {v: k for k, v in PR.items()}
cm = pd.read_csv(F('qual', '03_rider_code_matrix_v1.csv'), encoding='utf-8-sig').set_index('P')
code = lambda r, c: (cm.loc[RP[r], c] if r in RP and pd.notna(cm.loc[RP[r], c]) else '') if r in RP else 'n/a'

# ---------- profile ----------
on = M[M.onshift]; hr = pd.to_datetime(on.minute_unix + 7 * 3600, unit='s').dt.hour
night_share = ((hr >= 20) | (hr < 4)).groupby(on.pid).mean()
ev = PF[PF.period == 'evening']; mo = PF[PF.period == 'morning']
mo1 = mo.sort_values('reading_unix').groupby(['pid', 'workday']).first().reset_index()
g = D.groupby('pid')
prof = pd.DataFrame({'city': g.city.first(), 'n_workdays': g.size(), 'med_hours_shift': g.hours_shift.median(),
                     'med_return_clock': g.return_clock.median(), 'share_out_gt11h': g.out_h.apply(lambda s: (s.dropna() > 11).mean()),
                     'night_worker': night_share >= 0.4, 'med_ratio24': g.ratio24.median(), 'med_ratio_onshift': g.ratio_on.median(),
                     'med_ratio_night_home': g.ratio_night.median(), 'med_ratio_eve': g.ratio_eve.median()})
for c, nm in [('med_ratio24', 'rank24'), ('med_ratio_onshift', 'rank_onshift'), ('med_ratio_night_home', 'rank_night')]:
    prof[nm] = prof.groupby('city')[c].rank(ascending=False).astype('Int64')
prof['eve_minus_morn_pp'] = pp['mean']; prof['eve_share_lt80'] = ev.groupby('pid').pct_pb.apply(lambda s: (s < 80).mean())
prof['sd_morning_pctpb'] = mo1.groupby('pid').pct_pb.std(); prof['pvt_hours_slope'] = pe.b_hos
prof['P'] = [RP.get(r, '') for r in prof.index]
CODES = ['T_MID_HARD', 'ALERT_HAZE', 'ROAD_SLIP', 'SYM_BREATH', 'PREEXIST', 'PEF_EVE_LOWER', 'PEF_ATTR_FATIGUE', 'PEF_ATTR_DUST', 'HOME_PROTECT', 'AVOID_TACTIC', 'PRESS_CANT', 'CANT_SENSE', 'PPE_USE', 'MASK_NONE']
for c in CODES: prof[c] = [code(r, c) if r in RP else '' for r in prof.index]
prof.round(3).to_csv(T('rider_profile.csv'))
P('## profile'); P(prof.drop(columns=CODES).round(2).to_string())

# ---------- A1 ----------
P('\n## A1 long vs ~9 h riders')
D['onh'] = D.n_onshift / 60; D['dose_on'] = D.ratio_on * D.onh
r1 = D.groupby('pid').agg(city=('city', 'first'), hours=('hours_shift', 'median'), onh=('onh', 'median'), ratio_on=('ratio_on', 'median'), dose_on=('dose_on', 'median'), ratio24=('ratio24', 'median'))
r1['grp'] = np.where(r1.hours >= 11, 'long>=11h', np.where(r1.hours < 10, 'short<10h', 'mid'))
rows = []
for c, gg in r1.groupby('city'):
    L = gg[gg.grp == 'long>=11h']; S = gg[gg.grp == 'short<10h']
    if len(L) and len(S):
        gm = lambda s: np.exp(np.log(s).mean())
        tot = math.log(gm(L.dose_on) / gm(S.dose_on)); h = math.log(gm(L.onh) / gm(S.onh)); rr = math.log(gm(L.ratio_on) / gm(S.ratio_on))
        rows.append({'city': c, 'long': ','.join(L.index), 'short': ','.join(S.index), 'dose_ratio_long_vs_short': round(math.exp(tot), 2),
                     'share_from_hours': round(h / tot, 2), 'share_from_dirtier_hours': round(rr / tot, 2), 'ratio24_long_vs_short': round(gm(L.ratio24) / gm(S.ratio24), 2)})
A1 = pd.DataFrame(rows); A1.to_csv(T('a1_long_vs_short.csv'), index=False); P(A1.to_string(index=False)); P(r1.round(2).sort_values(['city', 'hours']).to_string())

# ---------- A2 ----------
P('\n## A2 evening drop vs typical shift length')
a2 = prof[['city', 'med_hours_shift', 'eve_minus_morn_pp']].dropna()
P('Spearman all', round(a2.med_hours_shift.rank().corr(a2.eve_minus_morn_pp.rank()), 2), {c: round(x.med_hours_shift.rank().corr(x.eve_minus_morn_pp.rank()), 2) for c, x in a2.groupby('city')}, 'n', len(a2))
# SVG scatter
W, H = 420, 320
xs, ys = a2.med_hours_shift, a2.eve_minus_morn_pp
sx = lambda v: 50 + (v - 4) / (13 - 4) * (W - 70); sy = lambda v: 20 + (3 - v) / (3 + 8) * (H - 60)
el = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" font-family="sans-serif" font-size="10"><rect width="100%" height="100%" fill="white"/>',
      f'<line x1="50" y1="{sy(0):.0f}" x2="{W-20}" y2="{sy(0):.0f}" stroke="#bbb"/>', f'<text x="50" y="{H-8}">median GPS hours on shift per workday (4-13)</text>',
      f'<text x="4" y="14">evening - morning %PB (pp), per rider (exploratory)</text>']
for r, row in a2.iterrows():
    colr = '#1f6fb4' if row.city == 'BKK' else '#d1495b'
    el.append(f'<circle cx="{sx(row.med_hours_shift):.1f}" cy="{sy(row.eve_minus_morn_pp):.1f}" r="4" fill="{colr}"/><text x="{sx(row.med_hours_shift)+5:.1f}" y="{sy(row.eve_minus_morn_pp)+3:.1f}">{r}</text>')
el.append(f'<text x="{W-120}" y="30" fill="#1f6fb4">BKK</text><text x="{W-80}" y="30" fill="#d1495b">CMI</text></svg>')
open(F('cases', 'fig_a2_drop_vs_hours.svg'), 'w').write('\n'.join(el))

# ---------- A3 ----------
P('\n## A3 hours in cleaner-than-city air')
a3 = D.groupby('pid').agg(city=('city', 'first'), clean_h_24=('min_clean_24h', lambda s: s.median() / 60), clean_h_on=('min_clean_onshift', lambda s: s.median() / 60), onh=('onh', 'median'))
a3['clean_share_on'] = a3.clean_h_on / a3.onh; a3.round(2).to_csv(T('a3_clean_hours.csv')); P(a3.round(2).sort_values(['city', 'clean_h_24']).to_string())

# ---------- A4 ----------
P('\n## A4 largest/smallest evening drops')
f4 = ev.assign(f4=tb(ev.F4_right_after_pvt)).groupby('pid').f4.mean()
a4 = prof[['city', 'eve_minus_morn_pp', 'med_hours_shift', 'med_return_clock', 'med_ratio_eve', 'rank24', 'P']].dropna(subset=['eve_minus_morn_pp']).copy()
a4['rank_eve'] = a4.groupby('city').med_ratio_eve.rank(ascending=False).astype(int); a4['share_eve_pef_right_after_pvt'] = f4
a4 = a4.sort_values('eve_minus_morn_pp'); sel = pd.concat([a4.head(3), a4.tail(3)]); sel.round(2).to_csv(T('a4_extreme_drops.csv')); P(sel.round(2).to_string())
P('cohort median share right after PVT', round(f4.median(), 2))

# ---------- A5 ----------
P('\n## A5 night workers vs same-city day workers')
v3 = V[~V.excl & V.y1_speed_rt100.notna()]
o13 = v3[v3.order.isin([1, 3])].pivot_table(index=['pid', 'workday'], columns='order', values='y1_speed_rt100').dropna()
o13 = (100 * (o13[3] / o13[1] - 1)).groupby('pid').median()
slot3 = V[V.order == 3].groupby('pid').protocol_slot.value_counts(normalize=True).unstack().fillna(0)
a5 = prof[['city', 'night_worker']].join(Q2.drop(columns='city')).join(night_share.rename('share_on_20_04')).join(o13.rename('y1_3rd_vs_1st_pct')).join(slot3[['after_home', 'during_shift']].add_prefix('pvt3_'))
a5 = a5[a5.city == 'CMI'] if a5[a5.night_worker].city.eq('CMI').all() else a5
a5.round(2).to_csv(T('a5_night_workers.csv')); P(a5.round(2).to_string())

# ---------- B6/B8/B9/C12/C14/D15 view ----------
P('\n## interviewed riders: profile + codes')
iv = prof[prof.P != ''].copy(); P(iv[['P', 'city', 'med_hours_shift', 'med_return_clock', 'share_out_gt11h', 'rank24', 'rank_onshift', 'rank_night', 'med_ratio_eve', 'eve_minus_morn_pp', 'eve_share_lt80', 'sd_morning_pctpb'] + CODES].round(2).to_string())

# ---------- B7 ----------
P('\n## B7 per-rider slopes of daily evening-morning %PB on hours and own PM')
ev_last = ev.sort_values('reading_unix').groupby(['pid', 'workday']).last().reset_index()
dd = mo1[['pid', 'workday', 'pct_pb']].merge(ev_last[['pid', 'workday', 'pct_pb']], on=['pid', 'workday'], suffixes=('_m', '_e'))
dd['drop'] = dd.pct_pb_e - dd.pct_pb_m; dd = dd.merge(D[['pid', 'workday', 'hours_shift', 'pm_mean']], on=['pid', 'workday'])
rows = []
for p, gg in dd.groupby('pid'):
    gg = gg.dropna()
    if len(gg) < 10: continue
    z = lambda s: (s - s.mean()) / s.std()
    X = np.column_stack([np.ones(len(gg)), z(gg.hours_shift), z(gg.pm_mean)]); b, se, _ = ols(gg['drop'].values.astype(float), X)
    rows.append({'pid': p, 'P': RP.get(p, ''), 'n': len(gg), 'b_hours_per_sd': b[1], 'se_hours': se[1], 'b_ownpm_per_sd': b[2], 'se_pm': se[2],
                 'PEF_ATTR_FATIGUE': code(p, 'PEF_ATTR_FATIGUE'), 'PEF_ATTR_DUST': code(p, 'PEF_ATTR_DUST')})
B7 = pd.DataFrame(rows); B7.round(2).to_csv(T('b7_drop_slopes.csv'), index=False); P(B7.round(2).to_string(index=False))

# ---------- B10 ----------
P('\n## B10 own Y1 on own 5 dirtiest vs 5 cleanest workdays')
yd = v3.groupby(['pid', 'workday']).y1_speed_rt100.mean().reset_index().merge(D[['pid', 'workday', 'pm_mean']], on=['pid', 'workday'])
rows = []
for p, gg in yd.groupby('pid'):
    if len(gg) < 12: continue
    gg = gg.sort_values('pm_mean'); c5 = gg.head(5).y1_speed_rt100.mean(); d5 = gg.tail(5).y1_speed_rt100.mean()
    rows.append({'pid': p, 'P': RP.get(p, ''), 'n_days': len(gg), 'y1_dirty5_vs_clean5_pct': 100 * (d5 / c5 - 1), 'ALERT_HAZE': code(p, 'ALERT_HAZE'), 'ROAD_SLIP': code(p, 'ROAD_SLIP')})
B10 = pd.DataFrame(rows); B10.round(2).to_csv(T('b10_dirty_vs_clean_y1.csv'), index=False); P(B10.round(2).to_string(index=False))
P('median all', round(B10.y1_dirty5_vs_clean5_pct.median(), 2), '| ALERT_HAZE=Y median', round(B10[B10.ALERT_HAZE == 'Y'].y1_dirty5_vs_clean5_pct.median(), 2))

# ---------- B11 ----------
P('\n## B11 R16 around interview (Apr 2): cold reported')
r16 = D[D.pid == 'R16'][['workday', 'ratio24', 'hours_shift']].merge(mo1[mo1.pid == 'R16'][['workday', 'pct_pb']], on='workday', how='outer').merge(
    yd[yd.pid == 'R16'][['workday', 'y1_speed_rt100']], on='workday', how='outer').sort_values('workday')
r16['y1_pct_of_own_median'] = 100 * r16.y1_speed_rt100 / r16.y1_speed_rt100.median()
win = r16[(r16.workday >= '2026-03-16') & (r16.workday <= '2026-04-30')]
P('R16 own medians: morning %PB', round(r16.pct_pb.median(), 1)); P(win.drop(columns='y1_speed_rt100').round(2).to_string(index=False))
r16.to_csv(T('b11_r16_timeline.csv'), index=False)

# ---------- C13 ----------
P('\n## C13 hours and range on own 5 dirtiest vs 5 cleanest city-PM workdays')
homes = pd.read_csv(F('foundation', '_work', 'home_locations.csv')); homes = homes[homes.home.astype(str) == 'True']
rows = []
for p, gg in D.dropna(subset=['city_pm']).groupby('pid'):
    if len(gg) < 12: continue
    hm = homes[homes.pid == p]
    mm = M[(M.pid == p) & M.onshift & M.lat.notna()]
    lat0, lon0 = hm.lat.values[:, None], hm.lon.values[:, None]
    dist = np.hypot((mm.lat.values - lat0) * 111, (mm.lon.values - lon0) * 111 * np.cos(np.radians(lat0))).min(axis=0)
    rng = pd.Series(dist, index=mm.index).groupby(mm.workday).quantile(0.95)
    gg = gg.sort_values('city_pm'); c5, d5 = gg.head(5), gg.tail(5)
    rows.append({'pid': p, 'P': RP.get(p, ''), 'city': city[p], 'hours_clean5': c5.hours_shift.mean(), 'hours_dirty5': d5.hours_shift.mean(),
                 'range95_km_clean5': rng.reindex(c5.workday).mean(), 'range95_km_dirty5': rng.reindex(d5.workday).mean(),
                 'AVOID_TACTIC': code(p, 'AVOID_TACTIC'), 'PRESS_CANT': code(p, 'PRESS_CANT'), 'q1_hours_slope': Q1.b_hours_shift.get(p)})
C13 = pd.DataFrame(rows); C13['d_hours'] = C13.hours_dirty5 - C13.hours_clean5; C13['d_range_km'] = C13.range95_km_dirty5 - C13.range95_km_clean5
C13.round(2).to_csv(T('c13_avoid_dirty_days.csv'), index=False); P(C13.round(2).to_string(index=False))

# ---------- D15 ----------
P('\n## D15 most/least exposed per city (24 h rank)')
for c, gg in prof.groupby('city'):
    gg = gg.sort_values('rank24'); P(c, 'most', gg.index[0], gg.P.iloc[0], '| least', gg.index[-1], gg.P.iloc[-1])
    iv_ = gg[gg.P != '']; P(c, 'interviewed most', iv_.index[0], iv_.P.iloc[0], 'rank', iv_.rank24.iloc[0], '| interviewed least', iv_.index[-1], iv_.P.iloc[-1], 'rank', iv_.rank24.iloc[-1])

open(F('cases', '_work', 'part2_digest.txt'), 'w').write('\n'.join(OUT)); print('\n'.join(OUT))
