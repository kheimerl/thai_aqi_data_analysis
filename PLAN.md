# Evaluation plan: PM2.5 exposure vs. PVT performance (within-subjects)

Scope: `pvt.csv` + `aq_sensor.csv` only. `peakflow.csv` is set aside for now as a separate respiratory (not cognitive) study, using the same framework later.

## 1. Cohort & data-quality findings

- 23 riders (from `2025-2026 Rider-Sensor-Summary.xlsx`, `summary` sheet), after excluding `Preechai`, `adisorn`, `Patte` (confirmed research staff, not study riders) and trivial test entries (`Nussara`, `Rach`, `Rac`, `test`).
- AQ sensors run almost the entire deployment window (2025-12-11 → 2026-04-30). Same-day PM2.5 coverage for the real cohort is **98.1%** (1885/1922 sessions) — full-period, tight exposure windows are feasible, not just a subset of the study.
- **Gotcha**: sensors `6021`, `6022`, `6029`/`6032` were reassigned between two different riders mid-study (e.g. `6021` = Chaiyalit through 2026-02-26, then GAMPANAT from 2026-03-04). Joins must go through `(PVT user, sensor, date range)`, not sensor number alone.

## 2. Outcome metrics (PVT)

`stimulation_times` shows ~300s test duration and ~90–100 trials — an abbreviated PVT-B-style protocol, not the standard 10-minute PVT. Validate any lapse threshold against literature for this abbreviated form before using it (the classic 500ms cutoff may not transfer as-is).

- **Primary (pre-registered)**: response speed = 1/RT per session — handles the right skew visible in raw response times, standard in PVT literature.
- **Secondary**: median RT, lapse count (threshold TBD), error rate (`1 - percent_success`).

## 3. Exposure metric

- Raw **PM2.5 (µg/m³)** as a continuous variable — not a binned AQI category, to avoid losing statistical power to arbitrary breakpoints. Report results back in AQI terms for interpretability if useful, but don't model on the binned index.
- **Primary window: per-day, cumulative.** Exposure is expected to accumulate over the day rather than reflect only the most recent air — the primary exposure variable is the time-weighted mean PM2.5 from shift start (first AQ reading that calendar day for that rider) up to the test timestamp, not a short trailing window. Also compute the integrated dose (sum of concentration × sampling interval over the same span) as an alternate operationalization of "cumulative" for robustness — mean concentration and integrated dose can diverge when duration varies a lot rider-to-rider or day-to-day.
- **Secondary/sensitivity windows**: 1h and 4h trailing windows, to test whether the effect is actually driven by recent exposure rather than the day's accumulation — i.e. these now serve as the alternative hypothesis check, not the primary metric.
- **Cumulative exposure ≠ duration**: a same-day mean is less confounded with shift length than integrated dose is, but both need shift duration / time-on-task controlled separately in the model regardless — otherwise a "cumulative exposure" effect can't be distinguished from a "long shift → fatigue" effect.
- Co-model **temperature and humidity** from the same sensor — plausible cognitive-stressor confounders correlated with PM2.5 (Bangkok/Chiang Mai heat).

## 4. Statistical model — per-rider regression, meta-analytically combined (primary)

The study design deliberately spans both the haze and clear-air seasons, so each rider has substantial within-person PM2.5 variance — the usual power objection to fitting riders independently (too little spread in one person's own exposure to estimate their own slope) mostly doesn't apply here. So the primary analysis fits every rider's own model separately, with no between-rider term at all, then combines the per-rider estimates statistically rather than pooling raw sessions into one regression.

**Step 1 — per rider `i`, fit independently on only that rider's sessions:**
```
1/RT_td ~ PM2.5_td + temperature_td + humidity_td + shift_duration_td
          + session_number_t (practice effect) + time_of_day + day_of_week
```
Extract `β_PM2.5,i` and its standard error `SE_i`.

**Step 2 — combine the 23 `(β_i, SE_i)` pairs via random-effects meta-analysis** (DerSimonian-Laird or REML), not a plain average — this weights each rider's estimate by its precision (`1/SE_i²`) and reports:
- a pooled effect estimate — the headline number for this analysis,
- `τ²` / `I²` — how much the effect itself varies across riders, which is scientifically interesting on its own (are some riders more PM2.5-sensitive than others — by age, baseline health, route?),
- a forest plot of the 23 rider-level slopes as the primary visualization.

**Why this is primary**: it structurally cannot suffer from the between-subject confound discussed earlier — no step ever borrows information across riders to estimate the effect itself, since each rider's slope comes only from their own data — and the season-spanning design gives each rider enough PM2.5 spread for that individual estimate to be meaningful.

## 5. Statistical model — pooled mixed model (secondary cross-check)

Keep the previously-discussed pooled model as a secondary comparison, not the headline result. Use Mundlak / group-mean centering to isolate the within-subject effect from between-rider confounds:

```
PM2.5_it        = time-weighted mean PM2.5, shift start -> test time (cumulative, same day)
PM2.5_between_i = mean(PM2.5_it) for rider i across all their sessions
PM2.5_within_it = PM2.5_it − PM2.5_between_i

1/RT_itd ~ PM2.5_within_it + PM2.5_between_i
           + temperature_it + humidity_it
           + shift_duration_it + session_number_i (practice effect)
           + time_of_day + day_of_week
           + (1 | rider_i) + (1 | day_d:rider_i)
```

- `PM2.5_within`'s coefficient should be compared against the primary meta-analytic pooled estimate from §4 — agreement corroborates both; a meaningful disagreement is worth chasing down (e.g. something the mixed model's shared covariate estimation is picking up that fully independent per-rider models can't, or vice versa).
- Random intercept per `rider`, nested with a `day:rider` random effect, since multiple PVT sessions can share a day's exposure (non-independent within-day).
- `session_number` per rider controls for practice effects — PVT scores are known to improve with repeated administration, and riders have 90+ repeated sessions each.

## Results so far (as of 2026-07-29)

Pipeline (aq_sensor.csv split, linked dataset) and §4/§5 are implemented and run (`scripts/`). Linked dataset: 1922 PVT sessions, 22 riders, 91.3% same-day PM2.5 coverage.

- **§4 primary (per-rider + random-effects meta-analysis)**: SEs clustered by `test_date` (riders take 2-3 PVT sessions/shift, so same-day sessions aren't independent — clustering matters: it moved the pooled result from non-significant to significant). Pooled β = **−0.00064 Hz/µg/m³** (95% CI [−0.00125, −0.00003], z=−2.05, **significant**), I²=44.8% (real cross-rider heterogeneity — most riders trend negative, a few, notably `Panatda_6018`, trend clearly positive).
- **§5 secondary (pooled mixed model)**: common-slope model gives β = −0.00020 (not significant) — smaller and non-significant relative to §4. Attempted random-slope versions to test whether forcing a homogeneous slope explains the gap; both specifications were numerically unusable (one didn't converge, one converged to a degenerate fit with SE ~400x the coefficient). This non-convergence is itself informative: it's consistent with genuine slope heterogeneity across riders (Pesaran & Smith 1995 — a pooled/common-slope estimator isn't a consistent estimator of the average effect under heterogeneous slopes; the mean-group estimator §4 uses is). **Trust §4 as the headline result**; §5's common-slope model is a weaker, only directionally-consistent cross-check, not a contradiction.
- **Dropped covariate**: `day_of_week`/`is_weekend` removed from both models — all 12 weekend PVT sessions in the raw data have no same-day AQ readings (sensors appear off on weekends), so after `dropna` the modeling frame has zero weekend rows and the covariate is structurally constant (was causing a singular design matrix in the mixed model).
- **Known residual issue**: `shift_duration_min_sameday` and `hour_of_day` are highly correlated (r=0.988, since most riders start shifts around the same clock time) — the model can't cleanly separate "time since shift start" from "time of day" with this data. Doesn't implicate PM2.5 directly but makes those two coefficients individually untrustworthy.
- **§6 exposure-window sensitivity** (`scripts/fit_window_sensitivity.py`): effect strengthens monotonically with window length — 1h β=−0.00016 (n.s., I²=6.1%), 4h β=−0.00033 (n.s., I²=11.0%), same-day β=−0.00064 (**significant**, I²=44.8%). Recent air alone shows almost nothing; the full day's cumulative exposure carries the signal. Supports the §3 decision to make same-day cumulative exposure primary rather than a trailing window.
- **§6 nonlinearity check** (`scripts/fit_nonlinearity_check.py`): per-rider quadratic term (`I(pm25_within**2)`) meta-analyzed the same way as the primary analysis (not pooled — an earlier pooled quartile-dummy version was replaced after review, since it inherited the same common-effect limitation as §5's Model A). Pooled quadratic coefficient = 4.1e-6, 95% CI [-9.9e-6, 1.8e-5], not significant — no evidence of curvature, linear model remains reasonable. I²=59.8% (real heterogeneity in curvature across riders even though the pooled estimate is null).

## 6. Robustness / secondary checks

- **Heterogeneity check**: `τ²`/`I²` from the §4 meta-analysis already quantifies how much the PM2.5 effect varies rider-to-rider — report it, and flag if it's large enough that a single pooled number is misleading. Done: I²=44.8%, real heterogeneity, see above.
- **Nonlinearity**: PM2.5-cognition effects may not be linear (threshold/saturation plausible) — fit a spline or quartile-binned version per rider as a robustness check against the linear primary model. Done at the pooled level (per-rider was infeasible, too few sessions per quartile bin) — see above.
- **Exposure-window sensitivity**: report the primary cumulative same-day measure alongside the 1h/4h trailing-window alternatives — if performance tracks cumulative exposure rather than recent air, the cumulative measure should fit better/more stably than the short windows. Done — see above.
- **Multiple comparisons**: 1/RT is the pre-registered primary outcome; median RT / lapses / error rate are secondary/exploratory, reported without inflating claims from them. Not yet implemented.

## Next steps

1. ~~Split `aq_sensor.csv` by `Node`~~ — done (`scripts/split_aq_by_node.sh`).
2. ~~Build the linked dataset~~ — done (`scripts/build_pvt_aq_dataset.py`).
3. ~~Fit §4 (primary) and §5 (secondary)~~ — done (`scripts/fit_primary_analysis.py`, `scripts/fit_pooled_mixed_model.py`).
4. ~~§6 robustness checks: exposure-window sensitivity, nonlinearity~~ — done (`scripts/fit_window_sensitivity.py`, `scripts/fit_nonlinearity_check.py`). See "Results so far" above.
5. Remaining: secondary outcome metrics (median RT, lapse count once a protocol-appropriate threshold is validated, error rate) run through the same primary pipeline, reported as exploratory only.
6. `peakflow.csv` (respiratory, secondary study) using the same framework, once cognitive analysis is considered settled.
