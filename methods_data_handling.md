# How the data are turned into numbers, how validity is checked, and what to improve

*Nov 10, 2026, coordinator. Explains `scripts/foundation_aq_minutes.py` and `scripts/build_pvt_aq_dataset_v4.py` (S1's v4 build, which S3 and S4 use), lists the validity checks, and recommends changes. Planning and analysis notes, not paper prose.*

---

## 1. From raw sensor rows to the numbers in the models

### 1.1 PM2.5 exposure
| Step | What the code does | Note |
|---|---|---|
| Raw | Each sensor logs about once per second: `Timestamp` (Unix), PM2.5, temperature, humidity, GPS. | 5 GB, 27 nodes |
| Clean | PM kept if 0–500; anything else is set to missing. GPS kept only inside a Thailand box (drops 0/0 fixes). | Clock check: `Datetime(UTC+7)` = `Timestamp` + 7 h on 1.25 M sampled rows, 0 mismatches |
| Minute | Per node per minute: sum and count of valid PM readings. | `foundation/_work/` |
| **Window value** | **Mean** = (sum of all valid readings in the window) / (number of valid readings). Each reading counts equally, so with regular 1-s sampling this is a time-weighted mean. **Not the median.** | `pm_*` columns |
| Coverage | `*_nmin` = minutes in the window with any PM. Stored, but **no minimum is enforced** (see §3c). | |

**Windows** (t0 = PVT start = logged end − 5 min, or the peak-flow reading time; t1 = PVT end):

| Column | Window | Used for |
|---|---|---|
| `pm_1h`, `pm_4h` | Mean over [t0 − 1 h, t0) and [t0 − 4 h, t0) | Acute exposure |
| `pm_sameday` | Mean from workday start (**04:00** local) to t0 | Primary "same-day" exposure |
| `pm_sameday_cum` | Sum of minute means / 60 over the same window (µg/m³·h) = mean × hours of data | Old v3 "cumulative dose" |
| `pm_after` | Mean from t1 to the end of the workday | **Placebo**: air after the test can't cause the test |
| `pm_sameday_v3def` | Midnight → logged test time | Only to reproduce v3 |
| `pm_regional_sameday` | Per hour, median of the *other* same-city riders' hourly means (≥ 2 riders, ≥ 30 min each), averaged over the person's own data hours | City background |
| `pm_local_sameday` | Own minus regional | Personal roadway increment |

### 1.2 PVT outcomes (per session of ~100 trials)
- Trials with RT < 100 ms are dropped as false starts (standard PVT convention).
- **Y1 mean speed** = mean of 1000/RT (Hz). Reciprocal RT is the recommended PVT metric because it is less skewed.
- **Y2 lapse %** = share of trials with RT ≥ 2 × the rider's own median RT. It is relative because phone touch latency varies by device; an absolute 500 ms cutoff would mean different things on different phones.
- **Y3 slowest 10%** = mean speed of the slowest decile of trials.

### 1.3 Peak flow
- One value per reading (L/min). **Personal best (PB)** = the person's 95th percentile of their own readings ≥ 50 L/min. **%PB** = reading / PB × 100.
- Period: morning 04:00–13:59, evening 15:00–03:59.

### 1.4 Models
Per rider: OLS of outcome on exposure plus covariates, with standard errors clustered by test day. Per-rider slopes are then pooled with a random-effects meta-analysis (DerSimonian–Laird, with Hartung–Knapp–Sidik–Jonkman (HKSJ) intervals as the headline because k is ~20). This "two-stage" or mean-group design (Pesaran & Smith 1995) compares each rider only with themselves. That removes stable between-person differences (age, lungs, phone, route) and lets slopes differ by person (gap analysis G1, G2, G11).

---

## 2. Why mean and not median (and when median matters)

- **Health effects follow the inhaled dose**, which is concentration × time. The time-weighted mean is the quantity that tracks dose, and it's what air-quality standards and epidemiology use (1-h, 24-h means). A median throws away short spikes, such as riding behind a bus, which are real dose for a rider.
- **But low-cost optical sensors also produce false spikes** (dust on the optics, condensation, start-up). A mean is sensitive to these.
- **Recommendation:** keep the **mean as primary** and add two pre-specified sensitivity versions:
  - A **median** of minute values in the same window.
  - A **clipped mean** (minute values capped at the node's own 99.9th percentile).

  If the three agree, spikes don't drive the result. Add one exploratory **peak metric** (maximum 5-min mean in the window) for the "spike" question.

---

## 3. Validity checks already in place, and problems found today

### 3.1 Already in place
| Check | Result |
|---|---|
| Clock (PVT/PEF local vs AQ Unix) | Explicit −7 h; GPS speed dips before tests only under the Bangkok clock; independent rebuild reproduces it (Gate 0) |
| Cohort | Riders defined by the mapping, whitespace-stripped handles, handovers split by date, outside-enrolment sessions flagged |
| Exposure range and duplicates | 0–500 kept; exact duplicates flagged |
| Confounding by person | Within-rider design (each rider is their own control) |
| Confounding by day | Placebo exposure (PM after the test), lead-2 placebo, within-week fixed effects |
| Small k | HKSJ intervals; Holm across the confirmatory family; model count reported |
| QC | Exclusion and descriptive flags fixed before linkage (`foundation/QC_RULES.md`) |

### 3.2 Problems found today (descriptive checks only, no outcomes)
- **a. "Cumulative dose" mostly measures time.** `pm_sameday_cum` correlates r = 0.45 with minutes of data; the mean correlates −0.07. Cumulative dose therefore carries time on shift (fatigue) with it. The old headline used it. **Demote it to secondary** and use mean concentration plus hours on shift as separate terms (already in the S3/S4 briefs).
- **b. The same-day window starts at 04:00 and includes time at home before riding.** That dilutes on-road exposure. **Add an on-shift mean** (GPS shift start → test start) as a pre-specified alternative. For riders, it is the exposure the job causes.
- **c. No minimum coverage rule.** Coverage is good (1-h window ≥ 50% filled for 94.5% of rider tests; 2.7% empty; same-day < 60 min for 0.5%; after-test < 60 min for 8.4%), but the rule must be written down. **Proposed:** 1 h ≥ 30 min, 4 h ≥ 120 min, same-day ≥ 60 min, after ≥ 60 min. Otherwise the value is missing.
- **d. The sensor's humidity reading is probably not ambient.** The rider-hour median is 43%, and **no rider-hour exceeds 85%**, which is implausible outdoors in Thailand. It is likely measured inside the sensor housing. **Tell AIT now:** a humidity correction will need external humidity (a meteorological station), not this column. Until then, don't use `rh_*` as a covariate or a humidity flag.
- **e. Peak flow is mostly a single blow.** Only ~17 sittings have repeat blows, while clinical guidance takes the best of three. Single blows add effort noise and bias towards lower values. This is a limitation, and S1's effort flags can't fix it.
- **f. Phone changes would shift RT.** A new phone can step-change touch latency. **Ask the team** whether riders changed phones. S3 can add a per-rider step-change check on median RT over time; it is exposure-free.

---

## 4. Better methods from the literature (gap analysis rows in brackets) and what we adopt

| Idea | Source / rationale | Adopt? |
|---|---|---|
| **Equivalence test for the null** (two one-sided tests, TOST; Lakens 2017) with a smallest effect of interest fixed in advance | Turns "not significant" into "effects larger than X are ruled out", which a reviewer can't dismiss as low power (G10, G11) | **Yes, S3 F1 and S4 L1.** Proposed smallest effect of interest: 0.2 within-rider SD of the outcome per +1 SD PM. Agree it with the professor before the run. |
| **Time-stratified comparison** (case-crossover logic: compare each test only with the same rider's tests in the same calendar month) | Standard in air-pollution epidemiology for seasonal and trend confounding (G5) | **Yes, as a sensitivity analysis** (rider × month fixed effects), next to the existing within-week FE |
| Distributed-lag models (e.g. DLNM) | Lagged and cumulative effects (G8) | **No:** the sensor is off between test days, so there's no lag history. State it as a limitation. |
| Mixed models with random slopes | Common in the cited panels (G1, G2) | Keep two-stage as primary (random-slope fits didn't converge before). Mixed model only as a check if it converges. |
| Personal vs station exposure | Personal monitors misclassify less than stations (Chatzidiakou 2019; G6) | Strength of the design. Remaining classical error **biases toward the null**: say so when reporting nulls. |
| Humidity and drift correction | Bulot 2019; Hollo 2026; Huang 2022 (G7) | **AIT.** Tell AIT the onboard RH looks internal (§3.2d). Ask whether they correct drift by month. |
| Negative-control exposure | Lipsitch et al. 2010 | Already used (placebo PM after the test) |
| Known-groups validity of the outcomes | If the instrument works, known differences should appear: PVT slower with age; peak flow lower in women and with age | **Yes:** the Bangkok pilot (§5) |
| Spike-robust exposure | Low-cost sensor artefacts | **Yes:** median and clipped-mean sensitivities (§2) |

---

## 5. Bangkok demographic pilot (new; `data/bkk-rider-info.txt`, granted Nov 10)

**Data:** sex and birth year for the 10 Bangkok riders, matched by name to R01–R10. The two files list riders 7 and 8 in different orders, so matching was by name, not row order. De-identified copy: `demographics/bkk_demographics.csv` (R##, sex, birth year, age in 2026), untracked. **5 men, 5 women, ages 21–56 (median 41).** Report ages in bands, never per rider with exact age: sex + birth year + city could identify someone.

**What a sample of 10 can and can't show:** a between-person comparison with n = 10 has almost no power for "significance". The pilot's real value is **measurement validity**, plus a first look at **who is more affected**.

| ID | Question | Analysis | Expected (known physiology) | Owner |
|---|---|---|---|---|
| B1 | Do PVT scores behave as expected? | Rider's mean speed (and median RT) vs age (Spearman ρ and slope, bootstrap CI over riders); men vs women | Slower with age; small or no sex difference | S3 |
| B2 | Does peak flow behave as expected? | Personal best by sex (ratio) and vs age | Men higher (often ~25–35%); declining with age. Without height, % predicted can't be computed. | S4 |
| B3 | Does the PM2.5 association differ by age or sex? | Meta-regression of the per-rider PM (z) slopes on age and sex, k = 10; within-Bangkok pooled slope | No prior direction; descriptive only | S3 (PVT), S4 (PEF) |
| B4 | Does the fatigue gradient differ by age? | Per-rider hours-on-shift slope vs age | Possibly steeper with age; descriptive | S3 |

**Rule for repeating in Chiang Mai** (fixed now, not after seeing results): repeat B1–B4 in Chiang Mai if **either** (i) B1 or B2 shows the expected direction (validity evidence is worth having for both cities, and the pooled n = 22 is the useful sample), **or** (ii) a B3/B4 meta-regression coefficient's 95% CI excludes 0. Either way, ask the team for Chiang Mai sex and birth year now. It's cheap, and analysis will need them for Table 1 anyway. **Height** (and weight) for all riders would allow % predicted peak flow; ask whether it was collected.
