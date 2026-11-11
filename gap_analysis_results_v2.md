# Gap Analysis v2: PM2.5 Exposure and PVT Performance in Platform Riders

*Prepared 2026-11-05. Builds on `data/gap/gap_analysis_results.md` (v1, left unchanged) and the analysis status in `PLAN.md` (results as of 2026-07-29 / 2026-08-07).*

---

## 0. What changed from v1, and the limits of this version

**What changed.** v1 marked every "Does your study address it?" cell **N/A** because the study protocol wasn't available. v2 scores those cells against the implemented pipeline described in `PLAN.md` §4–§6. It also adds a second literature sweep (Consensus, 2016–2026) aimed at the **analysis methods** of exposure–cognition field studies: how they handle between-person confounding, slope heterogeneity, practice effects, seasonality and sensor error. v1 focused on the topics those papers covered.

**Limits. Read these before reusing the matrix:**

1. **Future-work text comes from abstracts only.** Consensus returns abstracts, not full texts. Where an abstract states a limitation or future direction, it is quoted as a paraphrase. Where it doesn't, the cell says *"not in abstract"*. Nothing was reconstructed. Before citing a paper's "future work" in a manuscript, check the full text.
2. **Method details are as reported in abstracts.** If an abstract says "linear mixed-effects model" without saying whether exposure was person-mean-centered or whether test number was included, I say so. I don't assume either way.
3. **HCI-venue coverage for the core question is thin, and that is a finding in its own right.** Across five searches, the CHI / CSCW / IMWUT papers that came back were about *participatory air-quality sensing and display* (Hsu et al., 2017; Moore et al., 2018; Malliet et al., 2022). None estimated a **sensed-exposure → cognitive-performance** effect. Every quantitative exposure–cognition study retrieved was published in environmental-health, epidemiology or psychology venues. The matrix below therefore draws its methods comparisons from those venues and labels them that way, in line with v1's rule against relabeling.
4. **The "Methodological Contributions" section (§4) is an argument outline, not finished paper prose.** Organizational policy for this workspace doesn't allow me to draft substantial portions of academic papers. §4 sets out the claims, the supporting evidence from your robustness checks, the literature contrast and the caveats each paragraph has to carry, so you can write it in your own voice.

---

## 1. Study recap (from `PLAN.md`)

| Element | Implementation |
|---|---|
| Population | 22 analyzable Bangkok / Chiang Mai motorcycle rideshare riders (23 rostered; staff and test accounts excluded) |
| Period | 2025-12-11 → 2026-04-30, deliberately spanning the haze and clear-air seasons |
| Exposure | Per-second, rider-worn PM2.5 sensor. Primary metric: time-weighted mean from shift start to test time, same day. Alternatives: integrated dose; 1 h and 4 h trailing windows |
| Outcome | Abbreviated PVT (~300 s, ~90–100 trials). Primary outcome: 1/RT (response speed) |
| Linked data | 1,922 PVT sessions, 91.3% with same-day PM2.5 coverage |
| Primary model | Per-rider OLS (PM2.5 + temperature + humidity + shift duration + session_number + time of day), SEs clustered by test day, pooled by DerSimonian-Laird random-effects meta-analysis |
| Secondary model | Pooled mixed model with Mundlak within/between decomposition, `(1|rider)` only (the planned `(1|day:rider)` term was dropped in code because the fit was singular) |
| Headline | β = −0.00064 Hz per µg/m³ (95% CI −0.00125 to −0.00003; p = 0.040, DL + Wald-z), I² = 44.8% |
| Robustness | Exposure window, nonlinearity, learning effect (first week dropped), seasonal confound (session_number vs study_day), precision / funnel artifact, τ² estimator × HKSJ |
| Honest status | Point estimate is stable across specifications. **Significance is borderline** (p = 0.038–0.093 across τ² × inference combinations; all HKSJ p > 0.05) |

---

## 2. Literature extraction (new sweep, 2016–2026)

### 2a. Exposure → cognition field and panel studies (methods are the focus)

| Paper (venue) | RQ and population | Pooling / within-subject handling | Practice / learning handling | Future work / limitations (abstract only) |
|---|---|---|---|---|
| **Laurent et al., 2021** (*Environ. Res. Lett.*) [E1] | Are real-time indoor PM2.5 and CO2 associated with cognitive function? 302 office workers in 6 countries, **including Thailand**, followed for 12 months. Mobile Stroop and addition-subtraction tests. | Longitudinal repeated measures. The abstract doesn't specify the model or whether within- and between-person effects were separated. | Not in abstract (12 months of repeated tests makes this relevant). | Effects appear only above about 12 µg/m³ (threshold sensitivity analysis). Explicit future work not in abstract. |
| **Cleland et al., 2022** (*Environ. Health Perspect.*) [E2] | Are daily and hourly PM2.5 and wildfire smoke associated with attention? 10,228 US adults, 20 plays of a brain-training game. | Linear mixed-effects, longitudinal repeated measures. Exposure comes from **data fusion of monitoring networks, not personal sensing**. Subgroups stratified (age, gender, user behavior, region). | Not stated in abstract. A brain-training game is inherently subject to learning across the 20 plays. | "Further research is needed to elucidate these relationships." Subgroup differences not significant. |
| **Gignac et al., 2022** (*Environ. Int.*) [E3] | Is short-term NO2 associated with attention, stress and mood? 288 Barcelona adults over 14 days; daily Stroop by app (2,457 tests). | Linear regression with **random effects by participant** to estimate "intra-individual" associations. Adjusted for day of week and time-varying behaviors. Exposure is the **station mean or a location-based estimate**. | Not in abstract. | Not in abstract. |
| **Ke et al., 2022** (*Environ. Int.*) [E4] | Is short-term PM2.5 associated with executive function, and what is the EEG mechanism? 90 young adults, 4 one-week sessions, portable monitors. | Linear mixed-effects with lagged PM2.5. Only 4 test occasions per person. | Not in abstract. | Effect at lag 3 days, none at lag 0. Future work not in abstract. |
| **Zhang et al., 2021** (*Environ. Int.*) [E5] | Does physical activity offset the cognitive effects of PM2.5? Same cohort design as Ke et al. (n = 90). | Same 4-session panel. Model details not in abstract. | Not in abstract. | Not in abstract. |
| **Faherty et al., 2025** (*Nat. Commun.*) [E6] | Does acute PM exposure affect cognition, and through which inhalation pathway? 26 adults, controlled crossover. **Includes PVT.** | Within-subject experimental crossover. | Pre/post design within condition. | **PVT was *not* significantly affected by PM exposure**, while selective attention and emotion discrimination were. Important counterevidence for this study (see §3, G10). |
| **Shehab & Pope, 2019** (*Sci. Rep.*) [E7] | Does short-term PM exposure affect cognition? Double crossover (candle burning and roadside commuting). | Within-subject crossover. | Not in abstract. | Only some tests were affected. |
| **Ma et al., 2025** (*J. Health Econ.*) [E8] | Is contemporaneous PM2.5 associated with HCAP scores in Kenya? Portable monitors placed at testing sites. | Cross-sectional across individuals; binned exposure models. | Single administration. | Effect is larger in more educated people, so effect modification matters. |

### 2b. Cognitive assessment in uncontrolled settings (PVT and EMA)

| Paper (venue) | RQ and population | Methods relevant to this study | Future work / limitations (abstract only) |
|---|---|---|---|
| **Basner et al., 2017** (*SLEEP*) [C1] | Is the PVT really free of practice effects? | Repeated-administration analysis. | Challenges the "no practice effect" assumption: **PVT does show repeated-administration effects.** Directly relevant to the session_number finding. |
| **Grant et al., 2017** (*Behav. Res. Methods*) [C2] | Validation of 3-minute smartphone and tablet PVTs under sleep deprivation. | Validation against the standard PVT. | Supports brief PVT validity, but in a sleep-loss paradigm, not environmental exposure. |
| **Perzl et al., 2023** (*Assessment*) [C3] | What is the shortest reliable duration for in-the-wild DSST, SART and PVT? 46 adults over 2 weeks. | Within-person reliability of change. | A 30-second PVT is promising. v1 also noted that occupational samples are underrepresented. |
| **Sliwinski et al., 2016** (*Assessment*) [C4] | Reliability and validity of ambulatory cognitive assessment. | Foundational within-person reliability framework. | Not in abstract. |
| **Singh et al., 2022/2023** (*JMIR*) [C5] | Cognitive EMA reliability in clinical and community samples. | Within-person reliability ranged from 0.20 to 0.80. | Reliability varies by outcome. |
| **Shi et al., 2026** (*Arch. Phys. Med. Rehabil.*) [C6] | Feasibility, **learning effects** and stability of mobile cognitive assessments after stroke. | Longitudinal repeated in-vivo testing. | Learning effects are an explicit object of study. |
| **Fifield et al., 2025** (*Assessment*) [C7] / **Wrzus & Neubauer, 2021** (*Assessment*) [C8] | Meta-analyses of EMA compliance. | Compliance and missingness. | Missingness is common and under-reported. |

### 2c. Gig-worker health and platform labor

| Paper (venue) | RQ and population | Methods | Future work / gaps (abstract only) |
|---|---|---|---|
| **Tieanklin et al., 2024** (PACM HCI) [v1] | Do Thai motorcycle rideshare drivers consider pollution when deciding to drive? | Platform logs plus interviews. | **Explicit:** individual sensors to separate regional from acute roadway exposure. *This study is that follow-up.* |
| **Margerison et al., 2025** (*Am. J. Ind. Med.*) [W1] | Scoping review: 45 studies on gig work and health. | Review. | **Lacks prospective data, comparison groups and validated physical-health measures.** Causal evidence is lacking. |
| **Matilla-Santander et al., 2025** (*BMC Public Health*) [W2] | Scoping review: 40 studies. | Review. | Low sample sizes and missing control groups. |
| **Xu et al., 2026** (*Front. Public Health*) [W3] | Narrative review: 47 sources on Chinese delivery riders. | Review. | **Cross-sectional designs predominate, longitudinal evidence is limited, subgroup analyses are insufficient, and studies rely heavily on self-reported exposure.** |
| **Kwangsukstith et al., 2025** (*J. Occup. Med. Toxicol.*) [W4] | 709 motorcycle food-delivery riders in Chiang Mai. | Cross-sectional self-report survey. | 86.1% report PM exposure, but exposure is self-reported. Calls for health surveillance. |
| **Useche et al., 2024** (*Int. J. Occup. Saf. Ergon.*) [W5] | Does fatigue mediate the link between job stress and crashes? 248 riders. | Cross-sectional SEM. | Fatigue is "significant yet overlooked". |
| **Wen, 2026** (*Saf. Health Work*) [W6] | Working time, sleep and injury. 1,537 Shanghai riders. | Cross-sectional mediation. | Working hours are a direct exposure, which supports controlling for shift duration. |

### 2d. Air-quality sensing (HCI and sensor science)

| Paper (venue) | Relevance | Limitation / future work (abstract only) |
|---|---|---|
| **Moore et al., 2018** (**IMWUT**) [S1] | MAAV: multi-monitor PM2.5 sensing with annotation, 20–47-week deployments. | In-home, awareness-focused. No health outcomes. |
| **Hsu et al., 2017** (**CHI**) [S2] | Community-empowered air-quality monitoring for advocacy. | Evidence for regulators. No individual health linkage. |
| **Malliet et al., 2022** (**DIS**) [S3] | Reactivating citizen air-quality data through design. | Design-oriented. No exposure–outcome analysis. |
| **Chatzidiakou et al., 2019** (*Atmos. Meas. Tech.*) [S4] | Portable personal monitors have smaller error than fixed-station proxies. | Inaccurate personal exposure "severely limit[s] causal inference". |
| **Bulot et al., 2019** (*Sci. Rep.*) [S5] | Year-long field comparison of low-cost PM sensors. | Performance varies with PM source, concentration and **relative humidity**. |
| **Carter et al., 2023** (*ES&T Letters*) [S6] | Size-resolved accuracy of low-cost sensors. | Accurate mainly for PM1. PM2.5–10 fractions are biased and noisy. |
| **Hollo et al., 2026** (*ACS Environ. Au*) [S7] | Two-year co-location of 50 personal monitors. | Underestimation above 30 µg/m³, **temporal drift**, 5–11% of units failing. Humidity and dew-point correction cut RMSE by 42–50%. |
| **Huang et al., 2022** (*Sensors*) [S8] | Calibration of AirBeam2 sensors. | Bias under rain and high humidity. |
| **He et al., 2022** (*Atmos. Environ.*) [S9] / **Kim et al., 2021** (*Atmosphere*) [S10] | Personal monitors capture microenvironment spikes that station data miss. | Discrepancies are largest indoors and far from stations. |

---

## 3. Refined gap analysis matrix

**How to read the alignment rating:** it says how directly *the implemented pipeline* addresses the gap, given the evidence in `PLAN.md`. It is not a rating of the result's statistical strength. Several **High** rows still carry caveats that a reviewer will raise.

| # | Identified gap / methodological limitation | Source papers | How the literature currently handles it | How our pipeline handles it | Alignment |
|---|---|---|---|---|---|
| G1 | **Between-person confounding.** People with higher average exposure differ in other ways (age, route, baseline health), which biases pooled exposure slopes. | E1, E2, E3, E4/E5 | Mixed-effects models with random intercepts (E2, E4). E3 reports "intra-individual" associations via participant random effects. None of the abstracts reports person-mean centering. A random intercept alone does not separate within-person from between-person slopes unless exposure is decomposed. | Primary: **per-rider OLS**, so no between-rider information enters any slope; the 22 slopes are then meta-analyzed. Secondary: **Mundlak decomposition** (PM2.5_within vs PM2.5_between) as an explicit cross-check. | **High** |
| G2 | **Assuming one common slope.** Susceptibility likely varies across people, and a pooled common-slope estimator is inconsistent for the average effect when slopes differ (Pesaran & Smith, 1995). | E2 (stratified), E8 (effect modification), W3 (insufficient subgroup analysis) | A common slope plus stratified subgroups or interaction terms. Heterogeneity is usually tested as "subgroup differences: n.s." (E2). | Heterogeneity is quantified directly: **τ², I² = 44.8%**, and a forest plot of 22 slopes. A mean-group (meta-analytic) estimator is used. Random-slope mixed models failed to converge, which is consistent with real heterogeneity. | **High.** Caveat: k = 22 is too few to *explain* the heterogeneity with moderators. |
| G3 | **Practice and repeated-administration effects** inflating or masking exposure effects in high-frequency testing. | C1, C3, C4, C6, E2 (20 plays) | Often assumed absent for the PVT, an assumption C1 challenges. Elsewhere handled by run-in sessions, a test-index covariate, or not reported. | A **session_number** covariate in every per-rider model, plus a **first-week drop** check. The PM2.5 point estimate is stable (−0.00054 vs −0.00064, about 16% change). | **High** for protecting the PM2.5 estimate. Caveat: with the first week dropped, the CI includes zero. |
| G4 | **Long-horizon performance drift** (disengagement or fatigue over months), distinct from short-run practice. | C1, C7/C8 (compliance), W5 (fatigue) | Rarely studied. Most panels are 2 weeks (C3, E3) or 4 sessions (E4). | Found but not explained: session_number has a **negative** effect (β = −0.0045, z = −3.31, I² = 85.2%). Speed declines from 2.52 Hz in sessions 1–10 to 2.24 Hz in sessions 200+. | **Medium.** It is detected and shown to be separate from season (G5), but the mechanism is unknown. This is also a possible new finding in its own right. |
| G5 | **Seasonal and temporal confounding.** Exposure trends move together with calendar time and with test count. | E1 (12 months), E2 (wildfire season), S7 (sensor drift over time) | Day-of-week or season adjustment (E3). Collinearity between exposure trend and test index is rarely examined explicitly. | Three tests: (1) VIF for PM2.5 has a mean of 1.54; (2) swapping session_number for **study_day** gives β = −0.00067; (3) including both gives β = −0.00081, a *stronger* effect. | **High.** Caveat: no weekend data at all (sensors off), so day-of-week is not identifiable. |
| G6 | **Exposure misclassification from fixed stations or modeled exposure.** | E2, E3, S4, S9, S10, W3, W4, Tieanklin 2024 | Data fusion of monitoring networks (E2), station means (E3), site-level monitors (E8), or self-report (W3, W4). | A **rider-worn, per-second sensor**. Exposure is built from the shift start to the exact test timestamp. Same-day coverage is 91.3%. | **High** |
| G7 | **Low-cost sensor error:** humidity bias, drift, unit failure, saturation at high concentrations. | S5, S6, S7, S8 | Co-location calibration against reference instruments, plus humidity and dew-point correction models (S7: RMSE down 42–50%). | Temperature and humidity are **co-modeled as confounders**, not used to correct the sensor. `PLAN.md` documents **no reference co-location or calibration**. A constant per-sensor offset cancels out of within-rider slopes, but gain error and nonlinearity above ~30 µg/m³ (S7) do not. Classical measurement error attenuates estimates toward the null. Sensor reassignment between riders (6021, 6022, 6029/6032) adds unit-switch risk within the study window. | **Low.** This is the clearest gap in the pipeline (see §5). |
| G8 | **Choice of exposure window.** Effects may be acute, cumulative or lagged. | E2 (3 h), E3 (12/24 h), E4 (lag 3 days), E6 (4 h delay), E1 (real time) | One or a few windows chosen a priori. E4 found effects at a lag of 3 days only. | Same-day cumulative exposure (primary) vs 1 h and 4 h. The effect **strengthens monotonically** with window length (1 h n.s., 4 h n.s., same-day p = 0.040). | **High** for within-day windows. **Gap:** no multi-day lags (lag 1–3 days), which E4 suggests matter. |
| G9 | **Within-day non-independence** when several tests share one day's exposure. | E2, E3 (repeated daily tests) | Random effects by person. Day-level clustering is not reported in the abstracts. | **SEs clustered by test day** in the per-rider models (this moved the result from n.s. to significant). The secondary model only has a rider random intercept: `fit_pooled_mixed_model.py` dropped the `(1|day:rider)` term `PLAN.md` describes because it made the fit singular. | **High** (primary only) |
| G10 | **Whether the abbreviated PVT is sensitive to PM.** | C2, C3, **E6 (PVT null under controlled PM)**, E7 | Brief PVTs are validated mostly for sleep loss (C2), or for general in-the-wild reliability (C3). E6 found **no PVT effect** after controlled PM exposure. | ~300 s and ~90–100 trials, 1/RT as the primary outcome. Lapse threshold **not yet validated** for this format. | **Medium.** The design choice is defensible, but E6 is direct counterevidence that a reviewer will raise. Possible reconciliations to test: E6's exposure was a 1 h acute dose, while ours is cumulative over a day and over months; E6 had 26 lab participants, while we have about 1,900 field sessions. |
| G11 | **Small-k meta-analytic inference.** DerSimonian-Laird + Wald-z is anti-conservative at small k. | IntHout et al. 2014 (cited in `PLAN.md`; not in this sweep) | Rarely relevant in HCI, which seldom uses two-stage meta-analytic designs. | τ² estimators (DL, PM, REML) × inference (Wald-z, HKSJ): the point estimate varies by under 2%, but **p ranges from 0.038 to 0.093**, and every HKSJ result is above 0.05. | **Medium.** The pipeline is fully transparent, but the result is borderline. Report the HKSJ CI alongside DL. |
| G12 | **Precision artifacts:** imprecise individual slopes read as a "resistant subgroup". | No source in the retrieved literature (a standard meta-analysis concern) | No direct HCI or field precedent was found. | β–SE correlation r = 0.66, and SE tracks within-rider exposure SD (r = −0.74). Leaving out the highest-SE riders *strengthens* the estimate (z from −2.05 to −2.61). | **High.** A novel diagnostic for this literature. |
| G13 | **Time-on-task and workload confounding.** Long shifts mean both more exposure and more fatigue. | W5, W6, Mao et al. 2025, Tieanklin 2024 (platform logs) | Cross-sectional self-reported working hours (W6). | A shift-duration covariate. **However, r = 0.988 with hour of day**, so the two can't be separated. Platform workload (trip counts) is not modeled. | **Low–Medium.** The control exists but is effectively unidentified. |
| G14 | **Cross-sectional, self-reported worker-health evidence.** | W1, W2, W3, W4 | Surveys and qualitative studies dominate. | A **prospective, objective, high-frequency** within-person design over 4.5 months in the target workforce. | **High** for design. Caveat: n = 22 and no non-gig comparison group (W1's call for comparison groups remains open). |
| G15 | **Missingness and compliance** in field cognitive testing. | C7, C8 | Completion rates are reported; missingness mechanisms are seldom modeled. | 91.3% exposure coverage and complete-case analysis. Missingness of PVT sessions themselves (whether riders skip tests on bad-air or high-fatigue days) is not analyzed. | **Medium–Low** |
| G16 | **From measurement to worker agency and institutional action** (carried over from v1). | Tieanklin 2024, Dove 2022, Zhang 2022 (v1) | Participatory and advocacy tools; health-effect evaluation is rare. | Not addressed by the quantitative pipeline. | **Low.** Out of scope for this analysis; it needs a companion qualitative or design contribution. |

### Summary of positioning

- **Strongest, defensible differentiators:** G1, G2, G3, G5, G6, G9 and G12. The literature sampled here mostly uses mixed models with random intercepts, fixed-station or modeled exposure, few test occasions, and little reported attention to practice effects or slope heterogeneity. Your pipeline does all of these explicitly and shows its sensitivity checks.
- **Be careful about the framing "standard literature pools naively."** The sampled studies use mixed-effects models, which partially pool. That isn't naive. The accurate contrast is: *(a) common-slope vs heterogeneous-slope estimation, (b) random intercepts without within/between decomposition vs explicit isolation of within-person effects, and (c) personal vs modeled exposure.* Reviewers from environmental epidemiology will push back on stronger wording.
- **Weaknesses a reviewer will find:** G7 (no sensor calibration), G10 (the PVT null in E6), G11 (borderline inference) and G13 (shift-duration collinearity).

---

## 4. Methodological Contributions: argument outline

*This is an outline for you to write up; see §0.4. Each paragraph lists the claim, the evidence from your robustness checks, the literature contrast, and the caveat the paragraph must include to survive review.*

### Paragraph 1: Identification. Isolating the within-person exposure effect.

- **Claim:** In this design, estimating each rider's PM2.5 slope only from that rider's own data removes the between-person confounding that affects exposure–cognition panels.
- **Evidence from this study:** per-rider OLS feeding a random-effects meta-analysis (G1). The Mundlak secondary model gives a directionally consistent result. The season-spanning design gives each rider enough within-person exposure variance for this to work (the precision check shows SE ∝ 1/SD(exposure)).
- **Literature contrast:** prior panels use random-intercept mixed models with modeled or station exposure [E2, E3, E4]. Personal sensing is called for but rarely paired with cognitive outcomes [S4; Tieanklin 2024].
- **Extra point (G2):** I² = 44.8%, and the random-slope mixed models failed to converge. Under heterogeneous slopes, the common-slope estimator is not consistent for the average effect, while the mean-group estimator is (Pesaran & Smith, 1995). This explains why §5's −0.00020 is smaller than §4's −0.00064, and you can present it as a substantive finding rather than a discrepancy.
- **Required caveat:** k = 22 riders, and heterogeneity is not yet explained by moderators.

### Paragraph 2: Temporal validity. Practice, season and exposure window.

- **Claim:** The PM2.5 effect is not an artifact of learning, seasonal co-trends or window choice.
- **Evidence:** dropping the first week leaves the point estimate stable (G3). Swapping session_number for study_day, and including both, leaves PM2.5 stable or *stronger* (G5). The effect is monotonic across 1 h → 4 h → same-day windows (G8). Nonlinearity check: no curvature.
- **Literature contrast:** the PVT has been assumed free of practice effects, which C1 refutes. Most in-the-wild panels are short (2 weeks in C3 and E3; 4 sessions in E4) and don't report a practice-effect strategy in their abstracts.
- **Extra point (G4):** the *negative* session_number trend is a separate phenomenon, shown to be distinct from season. It is worth presenting as a field-specific observation (long-horizon decline) rather than hiding it.
- **Required caveat:** with the first week dropped, the CI includes zero, and multi-day lags (E4) are untested.

### Paragraph 3: Inferential transparency. Heterogeneity, precision and small-k inference.

- **Claim:** The pipeline reports how fragile the result is instead of presenting a single p-value.
- **Evidence:** precision diagnostics show that opposite-direction riders are imprecise estimates scattered around a real effect (G12). The τ² × inference grid shows a stable point estimate with borderline significance (G11).
- **Literature contrast:** no comparable sensitivity reporting appeared in the retrieved field exposure–cognition studies or in the HCI sensing papers.
- **Required caveat, stated plainly:** HKSJ p = 0.069–0.093. The defensible headline is "a consistent, modest negative association whose significance is borderline under small-k-appropriate inference." Pair this with G7 (uncalibrated sensors, which most likely bias toward the null) as a reason the true effect may be underestimated, but present that as a hypothesis, not a correction.

---

## 5. Recommended next steps, ranked by how much they would raise alignment

1. **G7 → calibration.** Co-locate the sensors, even post hoc, with a reference monitor or the nearest Thai PCD station. Apply a humidity correction (S7) and re-run the primary pipeline. If the effect strengthens, that supports the attenuation argument in §4 ¶3.
2. **G10 → engage E6 directly.** Add median RT and lapse outcomes (already planned in `PLAN.md` §6), and discuss why a cumulative real-world dose might affect the PVT when a 1 h lab dose does not.
3. **G8 → add lags of 1–3 days** to the window-sensitivity script, in line with E4.
4. **G13 → separate shift duration from clock time,** for example using riders with atypical start times, or by using hours since first sensor reading after a gap.
5. **G15 → a missingness check:** does same-day PM2.5 predict whether a scheduled PVT session is missing?
6. **G4 → characterize the session decline** (fatigue vs disengagement, e.g. by looking at the RT distribution's tail vs its median).

---

## References (new in v2; v1 references remain in `data/gap/gap_analysis_results.md`)

All links are Consensus records returned by the 2026-11-05 searches.

- [E1] [Associations between Acute Exposures to PM2.5 and Carbon Dioxide Indoors and Cognitive Function in Office Workers](https://consensus.app/papers/details/1c3cdb73ae755952a2aad25a0dc160f8/?utm_source=claude_desktop) (Laurent et al., 2021, *Environmental Research Letters*)
- [E2] [Short-Term Exposure to Wildfire Smoke and PM2.5 and Cognitive Performance in a Brain-Training Game](https://consensus.app/papers/details/0fa2dfec7ce35a1d9e90b6019d11e5e0/?utm_source=claude_desktop) (Cleland et al., 2022, *Environmental Health Perspectives*)
- [E3] [Short-term NO2 exposure and cognitive and mental health: A panel study based on a citizen science project in Barcelona](https://consensus.app/papers/details/f05e632b9e03558faeb4259dc0e868ec/?utm_source=claude_desktop) (Gignac et al., 2022, *Environment International*)
- [E4] [Short-term PM2.5 exposure and cognitive function: Association and neurophysiological mechanisms](https://consensus.app/papers/details/74ca06d07a815aebaeee26d32113d3de/?utm_source=claude_desktop) (Ke et al., 2022, *Environment International*)
- [E5] [Physical activity attenuates negative effects of short-term exposure to ambient air pollution on cognitive function](https://consensus.app/papers/details/7e6dfa9434cb580399be000ae3ded22d/?utm_source=claude_desktop) (Zhang et al., 2021, *Environment International*)
- [E6] [Acute particulate matter exposure diminishes executive cognitive functioning after four hours regardless of inhalation pathway](https://consensus.app/papers/details/1c4d625384e45294a68dfc878959c307/?utm_source=claude_desktop) (Faherty et al., 2025, *Nature Communications*)
- [E7] [Effects of short-term exposure to particulate matter air pollution on cognitive performance](https://consensus.app/papers/details/3ace7dbbaa2d50d99b507c1b258543de/?utm_source=claude_desktop) (Shehab & Pope, 2019, *Scientific Reports*)
- [E8] [Impacts of Contemporaneous Air Pollution Exposure on Cognitive Performance in Kenya](https://consensus.app/papers/details/46e95be2871c5a7381b0b64b9168a92d/?utm_source=claude_desktop) (Ma et al., 2025, *Journal of Health Economics*)
- [C1] [Repeated Administration Effects on Psychomotor Vigilance Test Performance](https://consensus.app/papers/details/c49115ad1f3a5f65826ad923b7e4b170/?utm_source=claude_desktop) (Basner et al., 2017, *SLEEP*)
- [C2] [3-minute smartphone-based and tablet-based psychomotor vigilance tests for the assessment of reduced alertness due to sleep deprivation](https://consensus.app/papers/details/af22da231c0754b3b5f9b4ceae3f5f3d/?utm_source=claude_desktop) (Grant et al., 2017, *Behavior Research Methods*)
- [C3] [Measuring Situational Cognitive Performance in the Wild](https://consensus.app/papers/details/221dceb8ecbc5f2d879a4630b428dce7/?utm_source=claude_desktop) (Perzl et al., 2023, *Assessment*)
- [C4] [Reliability and Validity of Ambulatory Cognitive Assessments](https://consensus.app/papers/details/c6dd72593de856888ef0f4bc533fef57/?utm_source=claude_desktop) (Sliwinski et al., 2016, *Assessment*)
- [C5] [Ecological Momentary Assessment of Cognition in Clinical and Community Samples](https://consensus.app/papers/details/706b67f434c954beaec23266dd4eb38c/?utm_source=claude_desktop) (Singh et al., 2022, *JMIR*)
- [C6] [Self-Administered Mobile Cognitive Assessments for Individuals With Stroke](https://consensus.app/papers/details/54d26f0ddc685eb289e8fcb48ce52900/?utm_source=claude_desktop) (Shi et al., 2026, *Arch. Phys. Med. Rehabil.*)
- [C7] [Completion Rates of Smart Technology EMA in Populations With a Higher Likelihood of Cognitive Impairment](https://consensus.app/papers/details/59a7f804b4cf5f2dbb4afedda57f86dc/?utm_source=claude_desktop) (Fifield et al., 2025, *Assessment*)
- [C8] [Ecological Momentary Assessment: A Meta-Analysis on Designs, Samples, and Compliance Across Research Fields](https://consensus.app/papers/details/ba205b2dfa595260b83f1cca8c2dd67a/?utm_source=claude_desktop) (Wrzus & Neubauer, 2021, *Assessment*)
- [W1] [Platform-Mediated Gig Work and Health: A Scoping Review](https://consensus.app/papers/details/b032ef7a36135b7f841facd94e91ced5/?utm_source=claude_desktop) (Margerison et al., 2025, *Am. J. Industrial Medicine*)
- [W2] [What is known about the health of location-based and online web-based digital labour platform workers?](https://consensus.app/papers/details/0979bd028b315d77963f9b0dcf16f7e0/?utm_source=claude_desktop) (Matilla-Santander et al., 2025, *BMC Public Health*)
- [W3] [Occupational injuries and health risks among food delivery riders in China](https://consensus.app/papers/details/6fd3228937b9523ea5c520c53a2f7584/?utm_source=claude_desktop) (Xu et al., 2026, *Frontiers in Public Health*)
- [W4] [Assessment of occupational hazards, health effects, and PPE usage among motorcycle food delivery riders in Thailand](https://consensus.app/papers/details/c9686431e3d75fb190dde102118a21c4/?utm_source=claude_desktop) (Kwangsukstith et al., 2025, *J. Occup. Med. Toxicol.*)
- [W5] [The hidden cost of your 'too fast food': stress-related factors and fatigue predict food delivery riders' occupational crashes](https://consensus.app/papers/details/6f33e5b07be75e938787132fbfa87410/?utm_source=claude_desktop) (Useche et al., 2024, *Int. J. Occup. Saf. Ergon.*)
- [W6] [Working Time Exposures and Occupational Injury Among Delivery Gig Workers in China](https://consensus.app/papers/details/939c42de50ff5d78993fca2f04e4798f/?utm_source=claude_desktop) (Wen, 2026, *Safety and Health at Work*)
- [S1] [Managing In-home Environments through Sensing, Annotating, and Visualizing Air Quality Data](https://consensus.app/papers/details/b223ed59e0035ceda057e02c89519ce3/?utm_source=claude_desktop) (Moore et al., 2018, *IMWUT*)
- [S2] [Community-Empowered Air Quality Monitoring System](https://consensus.app/papers/details/b8e6eecc31295b4dbf91a0788974ad92/?utm_source=claude_desktop) (Hsu et al., 2017, *CHI*)
- [S3] [The Air Quality Lens: Ambiguity as Opportunity to Reactivate Environmental Data](https://consensus.app/papers/details/7c3adcd065b156a5aed3d769cf0aec83/?utm_source=claude_desktop) (Malliet et al., 2022, *DIS*)
- [S4] [Characterising low-cost sensors in highly portable platforms to quantify personal exposure in diverse environments](https://consensus.app/papers/details/b5bd749716d6530cba27fbe755bb6128/?utm_source=claude_desktop) (Chatzidiakou et al., 2019, *Atmos. Meas. Tech.*)
- [S5] [Long-term field comparison of multiple low-cost particulate matter sensors in an outdoor urban environment](https://consensus.app/papers/details/f22266e9e94153eda3e9d2a738ec4a76/?utm_source=claude_desktop) (Bulot et al., 2019, *Scientific Reports*)
- [S6] [Size-Resolved Field Performance of Low-Cost Sensors for Particulate Matter Air Pollution](https://consensus.app/papers/details/04c3b7c70b905b4f9ce27ea7795085f2/?utm_source=claude_desktop) (Carter et al., 2023, *ES&T Letters*)
- [S7] [Validation and Correction of PM2.5 Mass Concentrations from Consumer-Grade Personal Exposure and Outdoor Fixed Monitors](https://consensus.app/papers/details/e570f6412ce85b89a60dd9c78081dd99/?utm_source=claude_desktop) (Hollo et al., 2026, *ACS Environmental Au*)
- [S8] [Field Evaluation and Calibration of Low-Cost Air Pollution Sensors for Environmental Exposure Research](https://consensus.app/papers/details/33ff4fb8c90850c0bba29b0c5a03c167/?utm_source=claude_desktop) (Huang et al., 2022, *Sensors*)
- [S9] [Network of low-cost air quality sensors for monitoring indoor, outdoor, and personal PM2.5 exposure in Seattle](https://consensus.app/papers/details/fef6718f7cf25a8693dd9d24b97e58e4/?utm_source=claude_desktop) (He et al., 2022, *Atmospheric Environment*)
- [S10] [Real-Time Low-Cost Personal Monitoring for Exposure to PM2.5 among Asthmatic Children](https://consensus.app/papers/details/ba1ae8a3124c5d50bb04c087f6e01c12/?utm_source=claude_desktop) (Kim et al., 2021, *Atmosphere*)
- Pesaran, M. H., & Smith, R. (1995). Estimating long-run relationships from dynamic heterogeneous panels. *Journal of Econometrics*, 68(1), 79–113. (Cited in `PLAN.md`; not retrieved in this sweep.)
- IntHout, J., Ioannidis, J. P. A., & Borm, G. F. (2014). The HKSJ method for random effects meta-analysis is straightforward and considerably outperforms the standard DerSimonian-Laird method. *BMC Medical Research Methodology*, 14, 25. (Cited in `PLAN.md`; not retrieved in this sweep.)

_Search results were found using Consensus (https://consensus.app). Upgrade to Consensus Pro to include more data like study design and key takeaways for every result.: https://consensus.app/pricing/?utm_source=claude_desktop_
