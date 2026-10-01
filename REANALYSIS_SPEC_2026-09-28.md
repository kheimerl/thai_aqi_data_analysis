# Corrected re-analysis spec — written 2026-09-28, BEFORE any corrected-exposure model was fit

> Copied into the repo on 2026-10-01; text below is unchanged from the working copy. That copy was last written 2026-09-28 14:45 PDT; the first corrected-exposure fit output was written at 14:49 PDT (file modification times).

Context: `build_pvt_aq_dataset.py` treats naive PVT timestamps (verified Bangkok-local: the
stop-to-test movement dip, 22.5% -> 8.0% moving, appears only under that reading) as UTC, and
builds day boundaries in the host timezone (America/Los_Angeles). Every exposure window in
`data/pvt_aq_linked.csv` therefore lies 3-7h after the test, or starts at 15:00 (14:00 after
2026-03-08) Bangkok time. Nothing below has been run on corrected exposure as of writing.

Plan of record: PLAN.md @ 078c788 (2026-07-29 11:38), the per-rider OLS + random-effects
meta-analysis committed before any fitting code existed (4d47a87, 12:00). Code of record:
`fit_primary_analysis.py` @ HEAD.

## R0 — plan of record, mechanical fixes only
- F1  Localize PVT timestamps to Asia/Bangkok.
- F2  All day boundaries in Asia/Bangkok.
- F3  Sessions with Bangkok clock time < 04:00 belong to the previous calendar day ("study day");
      for them hour_of_day = 24 + clock hour. Study day is also the clustering unit.
- F4  Primary window = [first AQ reading at/after 00:00 of study day, test time]. Sensors log
      24h/day on study days, so this is midnight->test (the plan's "first reading = shift start"
      premise is false; the literal operationalization is kept as primary, see S2).
- F5  temp/RH over the same window; shift_duration = minutes from window start to test (as coded).
- Unchanged: outcome = mean(1000/RT) over RT>0; formula
  `speed ~ pm25 + temp + rh + shift_duration + session_number + hour_of_day`;
  MIN_SESSIONS_PER_RIDER=15; CR1 by day; DerSimonian-Laird; Wald z.

## R1 — inference-corrected (THE HEADLINE if it disagrees with R0)
R0 with per-rider SEs = CR2 (Bell-McCaffrey) clustered by study day, pooled with REML tau^2 and
HKSJ t(k-1) inference.

## R2 — documented deviation: outcome cleaning
R1 with RT < 100 ms excluded as false starts (standard PVT practice; not in the plan).

## Secondary sensitivity (under R1)
- S1  Correctly aligned trailing 1h and 4h windows.
- S2  Shift-onset window: [first sustained movement of the study day, test]. Moving minute =
      100-2000 m GPS displacement; onset = first minute after 04:00 with >=3 moving minutes in the
      next 10. Sessions before onset are dropped from S2.

## Falsification (dominant threat: shared day-level / seasonal confounding)
- N1  Negative-control exposure: PM2.5 over (test, test+4h] under R1 — expected null. Also a joint
      per-rider model with pre-4h and post-4h together (a causal acute effect predicts pre dominates).
- N2  Pooled OLS: `speed ~ pm25_sameday + temp + rh + session_number + hour_of_day + C(rider)
      + C(region x study_day)`, SEs clustered by study day (two-way rider x day as a check). Keeps
      only same-day, same-city, cross-rider exposure differences.

## Decision rule
"Supported" requires: R1 significant at alpha=.05, AND N1 null with pre > post in the joint
model, AND N2 same sign with a CI that does not exclude the R1 estimate's neighborhood.
Anything else -> report as exploratory / not supported.
